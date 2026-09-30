"""PDF parser (pdfminer.six): text lines with typography, bookmarks as authoritative headings.

Cascade for structure: PDF outline (bookmarks) -> typography -> shape (handled in structure/).
Images: pages with (almost) no text layer are OCR'd whole; embedded pictures are cropped from the rendered
page and read by OCR / the vision model (vision.py). Without OCR installed, scanned files are `needs_ocr`.
Known limit (roadmap): no table extraction.
"""
from __future__ import annotations

import codecs
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import (LAParams, LTChar, LTCurve, LTFigure, LTImage, LTLine, LTRect, LTTextContainer,
                             LTTextLine)
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1

from ..config import Settings
from ..ir import IMAGE, LIST_ITEM, PARA, Block, ParsedDoc
from ..vision import ImageReader, get_reader, render_pdf_page

BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)
BULLET = re.compile(r"^\s*([•▪◦●■\-–*]|\d{1,2}[.)]|[a-z][.)])\s+\S")
PAGE_NUMBER = re.compile(r"^\s*(page\s+|pag\.?\s+|p\.\s*)?\d{1,4}(\s*(/|of|di)\s*\d{1,4})?\s*$", re.I)
EDGE_BAND = 0.08          # top/bottom fraction of the page considered header/footer area
REPEAT_SHARE = 0.4        # a line repeated on this share of pages in the edge band is boilerplate
MIN_CHARS_PER_PAGE = 20
VECTOR_MIN_ELEMENTS = 8   # a chart/diagram is many strokes; a lone rule or box is not
VECTOR_GAP_PT = 14        # strokes closer than this belong to the same drawing
VECTOR_PAD_PT = 8         # margin kept around a drawing so axis labels are inside the crop
DUPLICATE_SHARE = 0.6     # OCR text already present as native text in the crop adds nothing
FIGURE_MAX_SHARE = 0.85   # an "image" covering the page on a page that has text is a background, not a figure


@dataclass(frozen=True)
class _Figure:
    page: int
    bbox: tuple[float, float, float, float]   # pdfminer points, origin bottom-left
    page_width: float
    page_height: float
    vector: bool = False    # drawn with lines/curves (charts, diagrams) rather than an embedded bitmap

    @property
    def share(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, (x1 - x0) * (y1 - y0)) / max(1.0, self.page_width * self.page_height)


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    page: int
    page_height: float
    box: int
    x0: float = 0.0
    x1: float = 0.0


def _is_stamp(text: str) -> bool:
    """Vertical margin stamps (arXiv id, side watermarks) come out as one character per line."""
    return all(len(tok) == 1 for tok in text.split())


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def _line_features(line: LTTextLine) -> tuple[float, bool]:
    chars = [c for c in line if isinstance(c, LTChar) and c.get_text().strip()]
    if not chars:
        return 0.0, False
    size = Counter(round(c.size * 2) / 2 for c in chars).most_common(1)[0][0]
    bold = sum(bool(BOLD_FONT.search(c.fontname)) for c in chars) / len(chars) > 0.6
    return size, bold


def _strokes(el):
    if isinstance(el, (LTCurve, LTRect, LTLine)):
        yield el.bbox
    elif isinstance(el, LTFigure):
        for child in el:
            yield from _strokes(child)


def _vector_boxes(page, number: int) -> list[_Figure]:
    """Cluster the page's drawing strokes into figure regions (union-find on padded overlap)."""
    boxes = [b for el in page for b in _strokes(el)
             if (b[2] - b[0]) * (b[3] - b[1]) < 0.9 * page.width * page.height]   # ignore page frames
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a[0] - VECTOR_GAP_PT <= b[2] and b[0] - VECTOR_GAP_PT <= a[2] and \
               a[1] - VECTOR_GAP_PT <= b[3] and b[1] - VECTOR_GAP_PT <= a[3]:
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    figures = []
    for members in groups.values():
        if len(members) < VECTOR_MIN_ELEMENTS:
            continue
        x0, y0 = min(b[0] for b in members) - VECTOR_PAD_PT, min(b[1] for b in members) - VECTOR_PAD_PT
        x1, y1 = max(b[2] for b in members) + VECTOR_PAD_PT, max(b[3] for b in members) + VECTOR_PAD_PT
        figures.append(_Figure(number, (max(0, x0), max(0, y0), min(page.width, x1), min(page.height, y1)),
                               page.width, page.height, vector=True))
    return figures


def _images_in(el):
    if isinstance(el, LTImage):
        yield el
    elif isinstance(el, LTFigure):
        for child in el:
            yield from _images_in(child)


def _extract_lines(path: Path, want_vector: bool = False) -> tuple[list[_Line], int, list[_Figure]]:
    lines: list[_Line] = []
    pages = 0
    figures: list[_Figure] = []
    box_id = 0
    for number, page in enumerate(extract_pages(str(path), laparams=LAParams()), 1):
        pages += 1
        if want_vector:
            figures.extend(_vector_boxes(page, number))
        for el in page:
            figures.extend(_Figure(number, img.bbox, page.width, page.height) for img in _images_in(el))
            if not isinstance(el, LTTextContainer):
                continue
            box_id += 1
            for line in el:
                if not isinstance(line, LTTextLine) or not line.get_text().strip() or _is_stamp(line.get_text()):
                    continue
                size, bold = _line_features(line)
                lines.append(_Line(line.get_text().strip(), size, bold, line.y0, line.y1,
                                   number, page.height, box_id, line.x0, line.x1))
    return lines, pages, figures


def _drop_boilerplate(lines: list[_Line], pages: int) -> list[_Line]:
    def in_edge(l: _Line) -> bool:
        return l.y1 > l.page_height * (1 - EDGE_BAND) or l.y0 < l.page_height * EDGE_BAND

    counts = Counter(_norm(l.text) for l in lines if in_edge(l))
    limit = max(2, REPEAT_SHARE * pages)
    return [l for l in lines
            if not (in_edge(l) and (counts[_norm(l.text)] >= limit or PAGE_NUMBER.match(l.text)))]


def _merge(lines: list[_Line]) -> list[Block]:
    """Lines of one layout box with the same typography become one paragraph block."""
    blocks: list[Block] = []
    group: list[_Line] = []

    def flush() -> None:
        if not group:
            return
        text = ""
        for l in group:
            text = text[:-1] + l.text if text.endswith("-") and l.text[:1].islower() else f"{text} {l.text}".strip()
        first = group[0]
        kind = LIST_ITEM if BULLET.match(first.text) else PARA
        blocks.append(Block(text, kind, size=first.size, bold=all(l.bold for l in group),
                            loc=f"p.{first.page}"))
        group.clear()

    for l in lines:
        if group and (l.box != group[-1].box or l.page != group[-1].page
                      or abs(l.size - group[-1].size) > 0.5 or l.bold != group[-1].bold
                      or BULLET.match(l.text)):
            flush()
        group.append(l)
    flush()
    return blocks


def _outline(path: Path) -> list[tuple[int, str]]:
    try:
        with open(path, "rb") as fh:
            doc = PDFDocument(PDFParser(fh))
            return [(lvl, title.strip()) for lvl, title, *_ in doc.get_outlines() if title]
    except (PDFNoOutlines, Exception):  # malformed outlines must not block ingestion
        return []


def _apply_outline(blocks: list[Block], outline: list[tuple[int, str]]) -> list[Block]:
    wanted = {_norm(t): lvl for lvl, t in outline}
    out = []
    for b in blocks:
        lvl = wanted.get(_norm(b.text))
        out.append(Block(b.text, PARA, level_hint=lvl, style="pdf-bookmark", size=b.size,
                         bold=b.bold, loc=b.loc) if lvl and len(b.text) < 200 else b)
    return out


def _page_of(block: Block) -> int:
    return int(block.loc.split(".")[-1]) if block.loc.startswith("p.") else 0


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.lower()))


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _native_words(lines: list[_Line], fig: _Figure) -> set[str]:
    """Words of the native text lines sitting inside a figure's box (its own labels)."""
    x0, y0, x1, y1 = fig.bbox
    inside = (l.text for l in lines if l.page == fig.page and x0 <= (l.x0 + l.x1) / 2 <= x1 and y0 <= (l.y0 + l.y1) / 2 <= y1)
    return _words(" ".join(inside))


def _figure_blocks(path: Path, figures: list[_Figure], skip_pages: set[int], reader: ImageReader,
                   settings: Settings, lines: list[_Line]) -> dict[int, list[Block]]:
    """Crop each meaningful picture / drawing out of the rendered page and read it."""
    bitmaps = [f for f in figures if not f.vector]
    wanted = [f for f in figures if f.page not in skip_pages and settings.ocr_min_figure_share <= f.share <= FIGURE_MAX_SHARE
              and not (f.vector and any(b.page == f.page and _overlaps(b.bbox, f.bbox) for b in bitmaps))]
    out: dict[int, list[Block]] = {}
    if not wanted:
        return out
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    scale, rendered = settings.ocr_dpi / 72, {}
    try:
        for fig in sorted(wanted, key=lambda f: (f.page, -f.bbox[3])):   # top of the page first
            if fig.page not in rendered:
                rendered.clear()                                      # keep one rendered page in memory
                rendered[fig.page] = render_pdf_page(pdf, fig.page - 1, settings.ocr_dpi)
            x0, y0, x1, y1 = fig.bbox
            crop = rendered[fig.page].crop((int(x0 * scale), int((fig.page_height - y1) * scale),
                                            int(x1 * scale), int((fig.page_height - y0) * scale)))
            if (read := reader.read_image(crop)):
                info, text = read
                if fig.vector and not info.caption:          # OCR-only: drop drawings whose text is already native
                    ocr_words = _words(info.ocr)
                    if not ocr_words or len(ocr_words & _native_words(lines, fig)) / len(ocr_words) >= DUPLICATE_SHARE:
                        continue
                out.setdefault(fig.page, []).append(
                    Block(text, IMAGE, loc=f"p.{fig.page}", meta={"image": info.sha}))
    finally:
        pdf.close()
    return out


def _scanned_page_blocks(path: Path, pages: list[int], reader: ImageReader, settings: Settings) -> dict[int, list[Block]]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    out: dict[int, list[Block]] = {}
    try:
        for number in pages:
            paragraphs = reader.ocr_paragraphs(render_pdf_page(pdf, number - 1, settings.ocr_dpi))
            out[number] = [Block(text, LIST_ITEM if BULLET.match(text) else PARA, loc=f"p.{number}")
                           for text in paragraphs]
    finally:
        pdf.close()
    return out


UNUSABLE_TITLE = re.compile(r"microsoft word|untitled|printmgr|\.(docx?|pdf|tex|indd|pptx?)\b|^scan_", re.I)


def _meta_title(path: Path) -> str:
    """The title stored in the PDF metadata when it is a real one (Wikipedia prints, publisher files), else ''."""
    try:
        with open(path, "rb") as fh:
            raw = resolve1((PDFDocument(PDFParser(fh)).info or [{}])[0].get("Title")) or b""
    except Exception:   # a broken info dictionary must not block ingestion
        return ""
    if not isinstance(raw, bytes):
        return ""
    utf16 = raw.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE))
    try:
        text = raw.decode("utf-16" if utf16 else "utf-8").strip()
    except UnicodeDecodeError:      # PDFDocEncoding is close to latin-1; dropping bytes would turn "Società" into "Societ"
        text = raw.decode("latin-1").strip()
    return "" if len(text) < 4 or UNUSABLE_TITLE.search(text) or not any(c.isalpha() for c in text) else text


def parse(path: Path, settings: Settings | None = None) -> ParsedDoc:
    settings = settings or Settings()
    reader = get_reader(settings) if settings.ocr != "off" else None
    lines, pages, figures = _extract_lines(path, want_vector=bool(reader and (reader.ocr_enabled or reader.vision_enabled)))
    ocr = bool(reader and reader.ocr_enabled)
    per_page = Counter(l.page for l in lines)
    scanned = [n for n in range(1, pages + 1) if per_page.get(n, 0) == 0 or
               sum(len(l.text) for l in lines if l.page == n) < settings.ocr_min_page_chars]
    if not ocr and pages and sum(len(l.text) for l in lines) / pages < MIN_CHARS_PER_PAGE:
        return ParsedDoc(path.stem, "pdf", (), ("no text layer found: OCR required",), needs_ocr=True)

    lines = _drop_boilerplate([l for l in lines if l.page not in scanned or not ocr], pages)
    native: dict[int, list[Block]] = {}
    for block in _merge(lines):
        native.setdefault(_page_of(block), []).append(block)
    outline = _outline(path)
    if outline:
        native = {p: _apply_outline(bs, outline) for p, bs in native.items()}

    ocr_blocks = _scanned_page_blocks(path, scanned, reader, settings) if ocr and scanned else {}
    figure_blocks = _figure_blocks(path, figures, set(scanned), reader, settings, lines) if ocr or reader and reader.vision_enabled else {}
    blocks: list[Block] = []
    for number in range(1, pages + 1):
        blocks += ocr_blocks.get(number) or native.get(number, [])
        blocks += figure_blocks.get(number, [])
    if not blocks and pages:
        return ParsedDoc(path.stem, "pdf", (), ("no text layer found and OCR read nothing",), needs_ocr=True)
    warnings = []
    if ocr_blocks:
        warnings.append(f"OCR used for {len(ocr_blocks)} page(s)")
    if figures and not ocr:
        warnings.append("embedded images were not read (install structrag[ocr])")
    warnings.append("tables are not extracted from PDFs yet") if figures else None
    return ParsedDoc(title=_meta_title(path) or path.stem, format="pdf", blocks=tuple(blocks), warnings=tuple(w for w in warnings if w))

"""PDF parser (pdfminer.six): text lines with typography, bookmarks as authoritative headings.

Cascade for structure: PDF outline (bookmarks) -> typography -> shape (handled in structure/).
Known limits (roadmap): no table extraction, no OCR. Scanned files are flagged `needs_ocr`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTFigure, LTImage, LTTextContainer, LTTextLine
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfparser import PDFParser

from ..ir import LIST_ITEM, PARA, Block, ParsedDoc

BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)
BULLET = re.compile(r"^\s*([•▪◦●■\-–*]|\d{1,2}[.)]|[a-z][.)])\s+\S")
PAGE_NUMBER = re.compile(r"^\s*(page\s+|pag\.?\s+|p\.\s*)?\d{1,4}(\s*(/|of|di)\s*\d{1,4})?\s*$", re.I)
EDGE_BAND = 0.08          # top/bottom fraction of the page considered header/footer area
REPEAT_SHARE = 0.4        # a line repeated on this share of pages in the edge band is boilerplate
MIN_CHARS_PER_PAGE = 20


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


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def _line_features(line: LTTextLine) -> tuple[float, bool]:
    chars = [c for c in line if isinstance(c, LTChar) and c.get_text().strip()]
    if not chars:
        return 0.0, False
    size = Counter(round(c.size * 2) / 2 for c in chars).most_common(1)[0][0]
    bold = sum(bool(BOLD_FONT.search(c.fontname)) for c in chars) / len(chars) > 0.6
    return size, bold


def _extract_lines(path: Path) -> tuple[list[_Line], int, int]:
    lines: list[_Line] = []
    pages = images = 0
    box_id = 0
    for number, page in enumerate(extract_pages(str(path), laparams=LAParams()), 1):
        pages += 1
        for el in page:
            if isinstance(el, LTFigure) or isinstance(el, LTImage):
                images += 1
            if not isinstance(el, LTTextContainer):
                continue
            box_id += 1
            for line in el:
                if not isinstance(line, LTTextLine) or not line.get_text().strip():
                    continue
                size, bold = _line_features(line)
                lines.append(_Line(line.get_text().strip(), size, bold, line.y0, line.y1,
                                   number, page.height, box_id))
    return lines, pages, images


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


def parse(path: Path) -> ParsedDoc:
    lines, pages, images = _extract_lines(path)
    chars = sum(len(l.text) for l in lines)
    if pages and chars / pages < MIN_CHARS_PER_PAGE:
        return ParsedDoc(path.stem, "pdf", (), ("no text layer found: OCR required",), needs_ocr=True)
    lines = _drop_boilerplate(lines, pages)
    blocks = _merge(lines)
    outline = _outline(path)
    if outline:
        blocks = _apply_outline(blocks, outline)
    warnings = ("tables are not extracted from PDFs yet",) if images else ()
    return ParsedDoc(title=path.stem, format="pdf", blocks=tuple(blocks), warnings=warnings)

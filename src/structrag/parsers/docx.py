"""DOCX parser: body order preserved, heading styles/outline levels, run typography, tables."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..config import Settings
from ..ir import IMAGE, LIST_ITEM, PARA, TABLE, Block, ParsedDoc
from ..vision import get_reader

HEADING_STYLE = re.compile(r"^(heading|titolo|überschrift|título|titre)\s*(\d+)$", re.I)
TITLE_STYLES = {"title", "titolo"}


def _style_size(par: Paragraph) -> float | None:
    for run in par.runs:
        if run.font.size:
            return run.font.size.pt
    style = par.style
    while style is not None:
        if style.font.size:
            return style.font.size.pt
        style = style.base_style
    return None


def _is_bold(par: Paragraph) -> bool:
    runs = [r for r in par.runs if r.text.strip()]
    style_bold = bool(par.style.font.bold) if par.style is not None else False
    if not runs:
        return style_bold
    return all(r.bold or (r.bold is None and style_bold) for r in runs)


def _outline_level(par: Paragraph) -> int | None:
    lvl = par._p.find(f"{qn('w:pPr')}/{qn('w:outlineLvl')}")
    if lvl is None:
        return None
    val = int(lvl.get(qn("w:val")))
    return val + 1 if val < 9 else None  # 9 = body text


def _paragraph_block(par: Paragraph) -> Block | None:
    text = par.text.strip()
    if not text:
        return None
    name = par.style.name if par.style is not None else ""
    m = HEADING_STYLE.match(name)
    hint = int(m.group(2)) if m else (1 if name.lower() in TITLE_STYLES else _outline_level(par))
    is_list = par._p.find(f"{qn('w:pPr')}/{qn('w:numPr')}") is not None or name.lower().startswith("list")
    kind = LIST_ITEM if is_list and hint is None else PARA
    return Block(text, kind, level_hint=hint, style=name, size=_style_size(par), bold=_is_bold(par))


def _cell_text(tc) -> str:
    return " ".join("".join(t.text or "" for t in p.iter(qn("w:t"))) for p in tc.findall(qn("w:p")))


def _table_block(table: Table) -> Block | None:
    rows: list[tuple[str, ...]] = []
    # Walk the XML rows/cells directly: `row.cells` raises on ragged tables (gridBefore/gridAfter, missing
    # cells), which real-world documents are full of. Merged cells are single <w:tc> elements here.
    for tr in table._tbl.findall(qn("w:tr")):
        cells = [" ".join(_cell_text(tc).split()) for tc in tr.findall(qn("w:tc"))]
        if any(cells):
            rows.append(tuple(cells))
    if not rows:
        return None
    return Block("\n".join(" | ".join(r) for r in rows), TABLE, rows=tuple(rows))


def _image_blocks(par_element, doc, reader, seen: set[str]) -> list[Block]:
    """Pictures anchored in a paragraph, in order. Repeated images (logos, letterheads) are read once."""
    out = []
    for blip in par_element.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        part = doc.part.related_parts.get(rid) if rid else None
        if part is None:
            continue
        alt = next((d.get("descr") or d.get("title") or "" for d in par_element.iter(qn("wp:docPr"))), "")
        read = reader.read_bytes(part.blob, alt)
        if read and read[0].sha not in seen:
            seen.add(read[0].sha)
            out.append(Block(read[1], IMAGE, meta={"image": read[0].sha}))
    return out


def parse(path: Path, settings: Settings | None = None) -> ParsedDoc:
    doc = Document(str(path))
    reader = get_reader(settings or Settings())
    read_images = reader.ocr_enabled or reader.vision_enabled
    blocks: list[Block] = []
    seen: set[str] = set()
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            block = _paragraph_block(Paragraph(child, doc))
            extra = _image_blocks(child, doc, reader, seen) if read_images else []
        elif child.tag == qn("w:tbl"):
            block, extra = _table_block(Table(child, doc)), []
        else:
            continue
        if block:
            blocks.append(block)
        blocks.extend(extra)
    title = doc.core_properties.title or next(
        (b.text for b in blocks if b.level_hint == 1), path.stem)
    return ParsedDoc(title=title, format="docx", blocks=tuple(blocks))

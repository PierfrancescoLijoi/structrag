"""DOCX parser: body order preserved, heading styles/outline levels, run typography, tables."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..ir import LIST_ITEM, PARA, TABLE, Block, ParsedDoc

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


def _table_block(table: Table) -> Block | None:
    rows: list[tuple[str, ...]] = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:  # merged cells repeat the same _tc: keep the first only
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(tuple(cells))
    if not rows:
        return None
    return Block("\n".join(" | ".join(r) for r in rows), TABLE, rows=tuple(rows))


def parse(path: Path) -> ParsedDoc:
    doc = Document(str(path))
    blocks: list[Block] = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            block = _paragraph_block(Paragraph(child, doc))
        elif child.tag == qn("w:tbl"):
            block = _table_block(Table(child, doc))
        else:
            continue
        if block:
            blocks.append(block)
    title = doc.core_properties.title or next(
        (b.text for b in blocks if b.level_hint == 1), path.stem)
    return ParsedDoc(title=title, format="docx", blocks=tuple(blocks))

"""Intermediate representation shared by all parsers.

Parsers emit a flat list of Blocks with raw layout signals. Structure inference
(structure/) decides which blocks are headings; the tree builder then nests them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PARA, TABLE, LIST_ITEM = "para", "table", "list_item"


@dataclass(frozen=True)
class Block:
    text: str
    kind: str = PARA
    # Authoritative heading level from the format itself (md '#', docx 'Heading 2', pptx title).
    level_hint: int | None = None
    style: str | None = None        # format style name, e.g. "Heading 2", "Normal"
    size: float | None = None       # font size in points, when known
    bold: bool = False
    loc: str = ""                   # page / slide / sheet reference for citations
    rows: tuple[tuple[str, ...], ...] = ()   # table cells, header first when known
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def style_key(self) -> str:
        """Typography fingerprint used to learn per-layout heading rules."""
        size = round(self.size) if self.size else 0
        return f"{size}|{int(self.bold)}|{self.style or ''}"


def table_parts(block: Block) -> tuple[list[str], list[tuple[str, ...]], list[tuple[str, ...]]]:
    """(column names, preamble rows above the header, data rows) of a TABLE block."""
    idx = block.meta.get("header_row", 0)
    if idx < 0:   # header lives in an earlier block: every row is data
        return list(block.meta["column_names"]), [], list(block.rows)
    names = block.meta.get("column_names") or list(block.rows[idx])
    return list(names), list(block.rows[:idx]), list(block.rows[idx + 1:])


@dataclass(frozen=True)
class ParsedDoc:
    title: str
    format: str
    blocks: tuple[Block, ...]
    warnings: tuple[str, ...] = ()
    needs_ocr: bool = False


@dataclass
class Section:
    title: str
    level: int
    blocks: list[Block] = field(default_factory=list)   # own body blocks only
    children: list["Section"] = field(default_factory=list)
    summary: str = ""

    def walk(self, trail: tuple[str, ...] = ()):
        """Yield (section, breadcrumb) depth-first."""
        crumb = trail + ((self.title,) if self.title else ())
        yield self, crumb
        for child in self.children:
            yield from child.walk(crumb)

    @property
    def words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)

"""Section-scoped chunking: chunks never cross a section boundary and carry their breadcrumb.

Text is packed by paragraphs up to a word target; oversized paragraphs are split by sentence.
Tables are cut by rows with the header repeated, so every table chunk is self-describing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Settings
from .ir import IMAGE, LIST_ITEM, TABLE, Block, Section, table_parts

SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    ctx: str          # breadcrumb + text: what gets embedded and indexed
    kind: str         # "text" | "table" | "image"
    loc: str
    ref: str = ""     # image thumbnail id for kind == "image"


def _words(text: str) -> int:
    return len(text.split())


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an oversized paragraph by sentence, then hard-cut by words as a last resort."""
    parts, buf = [], []
    for sentence in SENTENCE.split(text):
        words = sentence.split()
        while len(words) > max_words:
            if buf:
                parts.append(" ".join(buf))
                buf = []
            parts.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if buf and _words(" ".join(buf)) + len(words) > max_words:
            parts.append(" ".join(buf))
            buf = []
        buf.extend(words)
    if buf:
        parts.append(" ".join(buf))
    return parts


def _md_row(cells: list[str] | tuple[str, ...]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _table_chunks(block: Block, settings: Settings) -> list[tuple[str, str]]:
    names, preamble, rows = table_parts(block)
    head = f"{_md_row(names)}\n{_md_row(['---'] * len(names))}"
    lead = ("\n".join(" ".join(c for c in r if c) for r in preamble) + "\n") if preamble else ""
    out, i = [], 0
    while i < len(rows) or (i == 0 and not rows):
        step = settings.table_rows_per_chunk
        while step > 1 and _words(" ".join(" ".join(r) for r in rows[i:i + step])) > settings.chunk_max_words:
            step //= 2   # wide rows: fewer rows per chunk to stay under the word cap
        batch = rows[i:i + step]
        body = "\n".join(_md_row(r) for r in batch)
        loc = f"{block.loc} rows {i + 1}-{i + len(batch)}" if rows else block.loc
        out.append(((lead if i == 0 else "") + head + ("\n" + body if body else ""), loc))
        i += max(step, 1)
        if not rows:
            break
    return out


def chunk_section(section: Section, crumb: tuple[str, ...], settings: Settings,
                  start: int = 0) -> list[Chunk]:
    prefix = " > ".join(crumb)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_loc = ""

    def emit(text: str, kind: str, loc: str, ref: str = "") -> None:
        ctx = f"{prefix}\n{text}" if prefix else text
        chunks.append(Chunk(start + len(chunks), text, ctx, kind, loc, ref))

    def flush() -> None:
        nonlocal buf, buf_loc
        if buf:
            emit("\n".join(buf), "text", buf_loc)
        buf, buf_loc = [], ""

    for block in section.blocks:
        if block.kind == IMAGE:      # a figure is its own chunk: its caption/OCR must not blur into the prose
            flush()
            pieces = _split_long(block.text, settings.chunk_max_words) if _words(block.text) > settings.chunk_max_words else [block.text]
            for piece in pieces:
                emit(piece, "image", block.loc, block.meta.get("image", ""))
            continue
        if block.kind == TABLE:
            flush()
            for text, loc in _table_chunks(block, settings):
                emit(text, "table", loc)
            continue
        line = f"- {block.text}" if block.kind == LIST_ITEM else block.text
        pieces = _split_long(line, settings.chunk_max_words) if _words(line) > settings.chunk_max_words else [line]
        for piece in pieces:
            if buf and _words(" ".join(buf)) + _words(piece) > settings.chunk_target_words:
                flush()
            buf_loc = buf_loc or block.loc
            buf.append(piece)
    flush()
    return chunks

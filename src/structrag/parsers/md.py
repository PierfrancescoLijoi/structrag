"""Markdown / plain-text parser: ATX + setext headings, fenced code, pipe tables, lists."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import Settings
from ..ir import IMAGE, LIST_ITEM, PARA, TABLE, Block, ParsedDoc
from ..vision import get_reader

ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SETEXT = re.compile(r"^(=+|-+)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
LIST = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)")
IMG = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+\"[^\"]*\")?\)\s*$")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


def _image_block(match: re.Match, base: Path, settings: Settings | None) -> Block | None:
    """Local images are read (OCR / vision); remote ones and unreadable ones fall back to their alt text."""
    alt, src = match.group("alt").strip(), match.group("src")
    target = (base / src).resolve() if "://" not in src else None
    reader = get_reader(settings or Settings())
    if target and target.is_file() and (reader.ocr_enabled or reader.vision_enabled):
        if (read := reader.read_bytes(target.read_bytes(), alt)):
            return Block(read[1], IMAGE, meta={"image": read[0].sha})
    return Block(f"[Figure] {alt}", PARA) if alt else None


def parse(path: Path, settings: Settings | None = None) -> ParsedDoc:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines and lines[0].strip() == "---":  # skip YAML front matter
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        lines = lines[end + 1:] if end else lines
    blocks: list[Block] = []
    para: list[str] = []
    in_code = False

    def flush() -> None:
        if para:
            blocks.append(Block(" ".join(s.strip() for s in para), PARA))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_code = not in_code
            if in_code:
                para.append(line)
            else:
                flush()
            i += 1
            continue
        if in_code:
            para.append(line)
            i += 1
            continue
        if (img := IMG.match(line)):
            flush()
            if (block := _image_block(img, path.parent, settings)):
                blocks.append(block)
            i += 1
            continue
        atx = ATX.match(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if atx:
            flush()
            blocks.append(Block(atx.group(2), level_hint=len(atx.group(1)), style="atx"))
        elif not para and line.strip() and SETEXT.match(nxt) and not LIST.match(line):
            blocks.append(Block(line.strip(), level_hint=1 if nxt.startswith("=") else 2, style="setext"))
            i += 1
        elif "|" in line and TABLE_SEP.match(nxt):
            flush()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i]:
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Block("\n".join(" | ".join(r) for r in rows), TABLE, rows=tuple(rows)))
            continue
        elif (m := LIST.match(line)):
            flush()
            blocks.append(Block(m.group(2), LIST_ITEM))
        elif not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    title = next((b.text for b in blocks if b.level_hint == 1), path.stem)
    return ParsedDoc(title=title, format="md", blocks=tuple(blocks))

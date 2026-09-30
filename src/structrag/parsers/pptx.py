"""PPTX parser: one level-1 section per slide, title placeholder = heading, notes included."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from ..config import Settings
from ..ir import IMAGE, LIST_ITEM, PARA, TABLE, Block, ParsedDoc
from ..vision import ImageReader, get_reader

TITLE_TYPES = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}


def _shapes(shapes):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _shapes(shape.shapes)
        else:
            yield shape


def _is_title(shape) -> bool:
    return shape.is_placeholder and shape.placeholder_format.type in TITLE_TYPES


def _picture_block(shape, loc: str, reader: ImageReader, seen: set[str]) -> Block | None:
    try:
        blob = shape.image.blob
    except (AttributeError, ValueError, KeyError):   # not a picture, or a linked (external) image
        return None
    alt = next(iter(shape._element.xpath(".//p:cNvPr/@descr")), "")
    read = reader.read_bytes(blob, alt)
    if not read or read[0].sha in seen:              # logos repeat on every slide: read them once
        return None
    seen.add(read[0].sha)
    return Block(read[1], IMAGE, loc=loc, meta={"image": read[0].sha})


def _slide_blocks(slide, number: int, reader: ImageReader, seen: set[str]) -> list[Block]:
    loc = f"slide {number}"
    shapes = sorted(_shapes(slide.shapes), key=lambda s: (s.top or 0, s.left or 0))
    title = next((s.text_frame.text.strip() for s in shapes if _is_title(s) and s.has_text_frame), "")
    blocks = [Block(title or f"Slide {number}", level_hint=1, style="slide-title", loc=loc,
                    meta={"untitled": not title})]
    for shape in shapes:
        if _is_title(shape):
            continue
        if shape.has_text_frame:
            pars = shape.text_frame.paragraphs
            for par in pars:
                text = "".join(r.text for r in par.runs).strip()
                if text:
                    kind = LIST_ITEM if par.level or len(pars) > 1 else PARA
                    blocks.append(Block(text, kind, loc=loc))
        elif (reader.ocr_enabled or reader.vision_enabled) and (pic := _picture_block(shape, loc, reader, seen)):
            blocks.append(pic)
        elif getattr(shape, "has_table", False) and shape.has_table:
            rows = tuple(tuple(" ".join(c.text.split()) for c in r.cells) for r in shape.table.rows)
            blocks.append(Block("\n".join(" | ".join(r) for r in rows), TABLE, rows=rows, loc=loc))
    if slide.has_notes_slide and (notes := slide.notes_slide.notes_text_frame.text.strip()):
        blocks.append(Block(f"Speaker notes: {notes}", PARA, loc=loc, meta={"notes": True}))
    return blocks


def parse(path: Path, settings: Settings | None = None) -> ParsedDoc:
    prs = Presentation(str(path))
    reader, seen = get_reader(settings or Settings()), set()
    blocks: list[Block] = []
    for number, slide in enumerate(prs.slides, 1):
        blocks.extend(_slide_blocks(slide, number, reader, seen))
    title = prs.core_properties.title or (blocks[0].text if blocks else path.stem)
    return ParsedDoc(title=title, format="pptx", blocks=tuple(blocks))

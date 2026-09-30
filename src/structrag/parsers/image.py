"""Standalone image files (.png .jpg .tif ...): OCR text and, if configured, a vision-model description."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..ir import IMAGE, PARA, Block, ParsedDoc
from ..vision import get_reader


def parse(path: Path, settings: Settings) -> ParsedDoc:
    reader = get_reader(settings)
    if not (reader.ocr_enabled or reader.vision_enabled):
        return ParsedDoc(path.stem, "image", (), ("OCR is off or not installed: image cannot be read",), needs_ocr=True)
    data = path.read_bytes()
    read = reader.read_bytes(data, alt=path.stem.replace("_", " ").replace("-", " "))
    if read is None:
        return ParsedDoc(path.stem, "image", (), ("no readable text or description found in the image",), needs_ocr=True)
    info, text = read
    return ParsedDoc(title=path.stem, format="image", warnings=(),
                     blocks=(Block(path.stem, PARA, level_hint=1, style="image-file", loc=path.name),
                             Block(text, IMAGE, loc=path.name, meta={"image": info.sha})))

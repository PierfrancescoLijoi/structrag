"""Parser registry: dispatch by extension after a cheap magic-bytes sanity check."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..ir import ParsedDoc

ZIP_MAGIC = b"PK\x03\x04"
PDF_MAGIC = b"%PDF"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED = {".md", ".markdown", ".txt", ".docx", ".pptx", ".xlsx", ".pdf"} | IMAGE_EXTS


class UnsupportedFormat(ValueError):
    pass


def _check_magic(path: Path, ext: str) -> None:
    with open(path, "rb") as fh:
        head = fh.read(4)
    if ext in (".docx", ".pptx", ".xlsx") and head != ZIP_MAGIC:
        raise UnsupportedFormat(f"{path.name}: not an OOXML file (legacy .doc/.ppt/.xls are not supported)")
    if ext == ".pdf" and head != PDF_MAGIC:
        raise UnsupportedFormat(f"{path.name}: not a PDF file")


def parse_file(path: Path, settings: Settings | None = None) -> ParsedDoc:
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFormat(f"unsupported extension '{ext}'")
    _check_magic(path, ext)
    settings = settings or Settings()
    if ext in IMAGE_EXTS:
        from . import image
        return image.parse(path, settings)
    if ext in (".md", ".markdown", ".txt"):
        from . import md
        return md.parse(path, settings)
    if ext == ".docx":
        from . import docx
        return docx.parse(path, settings)
    if ext == ".pptx":
        from . import pptx
        return pptx.parse(path, settings)
    if ext == ".xlsx":
        from . import xlsx
        return xlsx.parse(path, settings)
    from . import pdf
    return pdf.parse(path, settings)

"""Images -> searchable text: OCR (RapidOCR, PP-OCRv5, ONNX on CPU) and an optional local vision model.

Every image is keyed by the SHA-256 of its bytes: results are cached on disk, so re-ingesting an edited
document never re-reads unchanged images, and a thumbnail is kept so answers can show the figure they cite.
The text of a figure is `[Figure] <description>` + `Text in figure: <ocr>` and is indexed like any paragraph.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from .config import Settings

log = logging.getLogger(__name__)

MIN_SIDE_PX = 48                # bullets, rules, favicons: nothing worth indexing
MIN_AREA_PX = 6_000
MIN_OCR_SCORE = 0.5
THUMB_MAX_PX = 900
VISION_MAX_PX = 1024
VISION_TIMEOUT = 180.0
PARA_GAP = 0.9                  # OCR lines further apart than this many line-heights start a new paragraph
THUMB_QUALITY = 80

VISION_PROMPT = (
    "Describe this image for a search index in 2-4 factual sentences: what it shows (chart type, axes, "
    "trend, diagram parts, photo subject, table) and every visible number or label, quoted exactly. "
    "Describe only what is visible; never guess. If it contains only text, reply exactly: Text-only image.")


@dataclass(frozen=True)
class ImageInfo:
    sha: str                    # first 16 hex chars: also the thumbnail file name
    ocr: str = ""
    caption: str = ""

    def text(self, alt: str = "") -> str:
        parts = []
        description = self.caption if self.caption and self.caption.lower().rstrip(".") != "text-only image" else alt
        if description:
            parts.append(f"[Figure] {description}")
        if self.ocr:
            parts.append(f"{'' if parts else '[Figure] '}Text in figure: {self.ocr}")
        return "\n".join(parts)


class _Cache:
    """(sha, kind) -> text, in a tiny SQLite file next to the main database."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS img(sha TEXT, kind TEXT, value TEXT, PRIMARY KEY(sha, kind))")
        self.lock = threading.Lock()

    def get(self, sha: str, kind: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM img WHERE sha=? AND kind=?", (sha, kind)).fetchone()
        return row[0] if row else None

    def put(self, sha: str, kind: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO img VALUES(?,?,?)", (sha, kind, value))


class ImageReader:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cache = _Cache(settings.data_dir / "image_cache.sqlite")
        self.thumbs = settings.data_dir / "images"
        self._engine = None
        self._engine_lock = threading.Lock()
        self._ocr_missing = False

    # ---- capabilities -----------------------------------------------------------------------
    @property
    def ocr_enabled(self) -> bool:
        return self.s.ocr != "off" and not self._ocr_missing

    @property
    def vision_enabled(self) -> bool:
        return bool(self.s.vision_model)

    def _ocr_engine(self):
        with self._engine_lock:
            if self._engine is None:
                try:
                    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
                except ImportError:
                    log.warning("OCR needs: pip install 'structrag[ocr]' (images and scanned PDFs are skipped)")
                    self._ocr_missing = True
                    return None
                logging.getLogger("RapidOCR").setLevel(logging.WARNING)   # it logs every model load at INFO
                self._engine = RapidOCR(params={
                    "Rec.lang_type": LangRec(self.s.ocr_lang), "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                    "EngineConfig.onnxruntime.intra_op_num_threads": _threads(self.s)})
            return self._engine

    # ---- OCR ---------------------------------------------------------------------------------
    def _ocr_lines(self, image) -> list[tuple[float, float, str]]:
        """[(top_y, line_height, text)] in reading order, low-confidence lines dropped."""
        import numpy as np
        engine = self._ocr_engine()
        if engine is None:
            return []
        with self._engine_lock:   # one page at a time: keeps CPU use bounded
            result = engine(np.asarray(image.convert("RGB")))
        if result.boxes is None or result.txts is None:
            return []
        lines = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            if score >= MIN_OCR_SCORE and text.strip():
                ys = [p[1] for p in box]
                lines.append((min(ys), max(ys) - min(ys), text.strip()))
        return sorted(lines, key=lambda t: t[0])

    def ocr_paragraphs(self, image) -> list[str]:
        """Scanned page -> paragraphs (lines grouped by vertical gap)."""
        paragraphs: list[list[str]] = []
        prev_bottom = prev_height = None
        for top, height, text in self._ocr_lines(image):
            if prev_bottom is None or top - prev_bottom > PARA_GAP * max(prev_height, height, 1):
                paragraphs.append([])
            paragraphs[-1].append(text)
            prev_bottom, prev_height = top + height, height
        return [" ".join(p) for p in paragraphs]

    # ---- figures -----------------------------------------------------------------------------
    def read_bytes(self, data: bytes, alt: str = "") -> tuple[ImageInfo, str] | None:
        """(info, block text) for an embedded figure, or None when it is too small / has nothing to say."""
        from PIL import Image, UnidentifiedImageError
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("unreadable image skipped: %s", exc)
            return None
        return self.read_image(image, alt, hashlib.sha256(data).hexdigest())

    def read_image(self, image, alt: str = "", sha: str | None = None) -> tuple[ImageInfo, str] | None:
        w, h = image.size
        if min(w, h) < MIN_SIDE_PX or w * h < MIN_AREA_PX:
            return None
        if sha is None:
            sha = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        key = sha[:16]
        ocr = caption = ""
        if self.ocr_enabled:
            kind = f"ocr:{self.s.ocr_lang}"
            cached = self.cache.get(key, kind)
            if cached is None:
                cached = " ".join(t for _, _, t in self._ocr_lines(image))
                if not self._ocr_missing:
                    self.cache.put(key, kind, cached)
            ocr = cached
        if self.vision_enabled:
            kind = f"cap:{self.s.vision_model}"
            cached = self.cache.get(key, kind)
            if cached is None:
                cached = self._describe(image)
                if cached:                                  # never cache a failed call
                    self.cache.put(key, kind, cached)
            caption = cached or ""
        info = ImageInfo(key, ocr, caption)
        text = info.text(alt.strip())
        if not text:
            return None
        self._save_thumb(image, key)
        return info, text

    def _save_thumb(self, image, key: str) -> None:
        path = self.thumbs / f"{key}.jpg"
        if path.exists():
            return
        self.thumbs.mkdir(parents=True, exist_ok=True)
        thumb = image.convert("RGB")
        thumb.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
        thumb.save(path, "JPEG", quality=THUMB_QUALITY)

    # ---- vision model ------------------------------------------------------------------------
    def _describe(self, image) -> str:
        small = image.convert("RGB")
        small.thumbnail((VISION_MAX_PX, VISION_MAX_PX))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=85)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = {"model": self.s.vision_model, "max_tokens": 220, "temperature": 0,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": VISION_PROMPT},
                       {"type": "image_url", "image_url": {"url": uri}}]}]}
        try:
            r = httpx.post(f"{self.s.vision_base_url or self.s.base_url}/chat/completions", json=payload,
                           timeout=VISION_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            log.warning("vision model unavailable, figure indexed by OCR only: %s", exc)
            return ""


def _threads(settings: Settings) -> int:
    return settings.local_threads or max(1, min(6, (os.cpu_count() or 4) // 2))


@lru_cache(maxsize=4)
def get_reader(settings: Settings) -> ImageReader:
    """One reader (models, cache) per Settings; parsers call this instead of threading state through."""
    return ImageReader(settings)


def render_pdf_page(pdf, index: int, dpi: int):
    """PIL image of one PDF page (pypdfium2 document `pdf`, zero-based `index`)."""
    return pdf[index].render(scale=dpi / 72).to_pil().convert("RGB")

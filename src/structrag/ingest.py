"""Ingestion pipeline: parse -> infer structure (heuristics, then agent) -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .chunking import chunk_section
from .config import Settings
from .ir import TABLE, ParsedDoc
from .llm.client import ChatModel, Embedder
from .parsers import UnsupportedFormat, parse_file
from .pii import Detector, PiiMasker, load_detector
from .store import Store
from .structure.heuristics import Inference, dominant_style, infer
from .structure.profiles import OverrideStore, ProfileStore
from .structure.resolver import Decision, resolve_headings, resolve_table_headers
from .structure.tree import build_tree
from .summarize import document_summary, section_summary

log = logging.getLogger(__name__)
HASH_BLOCK = 1 << 20


@dataclass(frozen=True)
class IngestResult:
    status: str                      # ingested | duplicate | needs_ocr | failed
    doc_id: int | None = None
    n_sections: int = 0
    n_chunks: int = 0
    strategy: str = ""
    confidence: float = 0.0
    agent_decisions: int = 0
    queued_for_review: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""
    pii: dict = field(default_factory=dict)   # masked distinct values per label


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


def _apply_overrides(doc: ParsedDoc, inf: Inference, overrides: dict) -> Inference:
    """Human corrections win over heuristics: exact-text heading levels (0 = body)."""
    levels, ambiguous = list(inf.levels), list(inf.ambiguous)
    for idx, block in enumerate(doc.blocks):
        if block.text in overrides["headings"]:
            level = overrides["headings"][block.text]
            levels[idx] = level or None
            if idx in ambiguous:
                ambiguous.remove(idx)
    return replace(inf, levels=levels, ambiguous=ambiguous)


def _apply_header_overrides(doc: ParsedDoc, overrides: dict) -> ParsedDoc:
    blocks = list(doc.blocks)
    for idx, b in enumerate(blocks):
        rng = b.meta.get("range")
        if b.kind == TABLE and rng in overrides["headers"]:
            row = overrides["headers"][rng]
            blocks[idx] = replace(b, meta={**b.meta, "header_row": row, "column_names": list(b.rows[row]),
                                           "header_ambiguous": False})
    return replace(doc, blocks=tuple(blocks))


def _mask_chunk(c, masker: PiiMasker):
    """ctx = breadcrumb + text: mask each part once and rebuild, so both copies always agree."""
    prefix = c.ctx[:len(c.ctx) - len(c.text) - 1] if c.ctx != c.text else ""
    text = masker.mask(c.text)
    ctx = "\n".join((masker.mask(prefix), text)) if prefix else text
    return replace(c, text=text, ctx=ctx)


def _mask_section(section: dict, masker: PiiMasker) -> dict:
    chunks = [_mask_chunk(c, masker) for c in section["chunks"]]
    title, path, summary = (masker.mask(section[k]) for k in ("title", "path", "summary"))
    return {**section, "title": title, "path": path, "summary": summary, "chunks": chunks,
            "card": f"{path}\n{summary}"}


INDEX_VERSION = 1   # bump when parsing/chunking logic changes: sync then rebuilds stale documents


def embedder_id(settings: Settings) -> str:
    """Vectors from different models are incompatible: this id decides when embeddings can be reused."""
    return {"hash": "hash", "local": f"local:{settings.local_embed_model}"}.get(
        settings.embedder, f"server:{settings.embed_model}")


def index_signature(settings: Settings) -> str:
    """Everything besides file content that shapes the index: a change means existing documents are stale."""
    s = settings
    return "|".join(map(str, (INDEX_VERSION, embedder_id(s), s.chunk_target_words, s.chunk_max_words,
                              s.table_rows_per_chunk, s.pii_mode, s.pii_model if s.pii_mode == "mask" else "")))


class Ingestor:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder,
                 llm: ChatModel | None = None):
        self.s, self.store, self.embedder, self.llm = settings, store, embedder, llm
        self.embed_id, self.index_sig = embedder_id(settings), index_signature(settings)
        self._lock = threading.RLock()   # upload worker, watcher and API calls must not ingest concurrently
        self.profiles = ProfileStore(settings.profiles_dir)
        self.overrides = OverrideStore(settings.profiles_dir)
        self.pii_detector: Detector | None = load_detector(settings) if settings.pii_mode == "mask" else None

    def _embed(self, texts: list[str], reuse: dict) -> np.ndarray:
        """Embed only texts without a stored vector (unchanged chunks of an edited file are free)."""
        todo = [t for t in dict.fromkeys(texts) if t not in reuse]
        fresh = dict(zip(todo, self.embedder.embed(todo))) if todo else {}
        return np.vstack([reuse[t] if t in reuse else fresh[t] for t in texts]) if texts else np.zeros((0, 1))

    def ingest_file(self, path: Path, reuse: dict | None = None) -> IngestResult:
        with self._lock:
            return self._ingest_file(path, reuse)

    def _ingest_file(self, path: Path, reuse: dict | None) -> IngestResult:
        sha = file_sha256(path)
        if (existing := self.store.find_by_hash(sha)):
            return IngestResult("duplicate", existing["id"])
        try:
            doc = parse_file(path, self.s)
        except (UnsupportedFormat, Exception) as exc:   # a broken file must not stop a batch
            log.warning("parse failed for %s: %s", path.name, exc)
            return IngestResult("failed", error=str(exc))
        if doc.needs_ocr:
            meta = self._meta(path, sha, doc, "", 0.0, "needs_ocr", 0)
            doc_id = self.store.save_document(meta, None, [])
            return IngestResult("needs_ocr", doc_id, warnings=doc.warnings)
        return self._build(path, sha, doc, reuse or {})

    def _resolve_structure(self, doc: ParsedDoc, sha: str) -> tuple[ParsedDoc, Inference, list[Decision]]:
        overrides = self.overrides.get(sha)
        doc = _apply_header_overrides(doc, overrides)
        decisions: list[Decision] = []
        if self.llm:
            doc, table_decisions = resolve_table_headers(doc, self.llm, self.s)
            decisions += table_decisions
        body_key = dominant_style(doc.blocks)
        inf = infer(doc, self.profiles.get(doc.format, body_key))
        inf = _apply_overrides(doc, inf, overrides)
        if self.llm and inf.ambiguous:
            levels, heading_decisions = resolve_headings(doc, inf, self.llm, self.s)
            inf = replace(inf, levels=levels, ambiguous=[])
            decisions += heading_decisions
        return doc, inf, decisions

    def _learn_and_queue(self, doc_id: int, doc: ParsedDoc, inf: Inference,
                         decisions: list[Decision]) -> int:
        observations = [(d.style_key, d.level) for d in decisions if d.kind == "heading" and d.accepted]
        self.profiles.learn(doc.format, inf.body_key, observations)
        queued = 0
        for d in decisions:
            if d.accepted:
                continue
            self.store.queue_review(doc_id, {
                "kind": d.kind, "idx": d.idx, "text": d.text, "probs": d.probs, "options": d.options,
                "style_key": d.style_key, "level": d.level or 1, "body_key": inf.body_key,
                "format": doc.format})
            queued += 1
        return queued

    def _meta(self, path: Path, sha: str, doc: ParsedDoc, summary: str, conf: float, status: str,
              words: int, strategy: str = "", pii: dict | None = None) -> dict:
        return {"path": str(path), "sha256": sha, "format": doc.format, "title": doc.title,
                "summary": summary, "words": words, "strategy": strategy, "confidence": conf,
                "status": status, "warnings": list(doc.warnings), "pii": pii or {},
                **self._file_sig(path), "embed_id": self.embed_id, "index_sig": self.index_sig}

    @staticmethod
    def _file_sig(path: Path) -> dict:
        st = path.stat()
        return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}

    def _build(self, path: Path, sha: str, doc: ParsedDoc, reuse: dict) -> IngestResult:
        doc, inf, decisions = self._resolve_structure(doc, sha)
        tree = build_tree(doc.title, doc.blocks, inf.levels)
        words = sum(len(b.text.split()) for b in doc.blocks)
        sections: list[dict] = []
        for section, crumb in tree.walk():
            chunks = chunk_section(section, crumb, self.s)
            if not chunks and not section.children and section.level == 0:
                continue
            summary = section_summary(section, self.llm, self.s)
            sections.append({"level": section.level, "title": section.title or doc.title,
                             "path": " > ".join(crumb) or doc.title, "summary": summary,
                             "card": f"{' > '.join(crumb) or doc.title}\n{summary}", "chunks": chunks})
        if not any(s["chunks"] for s in sections):
            return IngestResult("failed", error="document has no extractable text", warnings=doc.warnings)

        doc_summary = document_summary(tree, words, self.llm, self.s)
        pii: dict = {}
        if self.pii_detector:
            masker = PiiMasker(self.pii_detector, self.s.pii_min_score)
            doc_summary, sections = masker.mask(doc_summary), [_mask_section(s, masker) for s in sections]
            doc = replace(doc, title=masker.mask(doc.title))
            pii = masker.counts
        doc_vec = self._embed([f"{doc.title}\n{doc_summary}"], reuse)[0]
        sec_vecs = self._embed([s["card"] for s in sections], reuse)
        flat = [c for s in sections for c in s["chunks"]]
        chunk_vecs = self._embed([c.ctx for c in flat], reuse)
        cursor = 0
        payload = []
        for s, sv in zip(sections, sec_vecs):
            chs = []
            for c in s["chunks"]:
                chs.append({"text": c.text, "ctx": c.ctx, "kind": c.kind, "loc": c.loc, "emb": chunk_vecs[cursor]})
                cursor += 1
            payload.append({**s, "emb": sv, "chunks": chs})

        pending = [d for d in decisions if not d.accepted]
        conf = inf.confidence if not decisions else max(inf.confidence, 0.85 if not pending else 0.5)
        status = "review" if pending else "ok"
        meta = self._meta(path, sha, doc, doc_summary, conf, status, words, inf.strategy, pii)
        doc_id = self.store.save_document(meta, doc_vec, payload)
        queued = self._learn_and_queue(doc_id, doc, inf, decisions)
        return IngestResult("ingested", doc_id, len(payload), len(flat), inf.strategy, conf,
                            len(decisions), queued, doc.warnings, pii=pii)

    def reingest(self, doc_id: int) -> IngestResult:
        with self._lock:
            return self._reingest(doc_id)

    def _reingest(self, doc_id: int) -> IngestResult:
        """Rebuild a document from its stored file (file edited, index stale, or human correction).

        An edited file is stored as a new version first and the old one removed only on success, so a
        broken edit never wipes the index. Vectors of unchanged chunks are reused (same embedding model)."""
        row = self.store.conn.execute(
            "SELECT path, sha256, embed_id FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row or not Path(row["path"]).exists():
            return IngestResult("failed", error="source file is no longer available")
        path = Path(row["path"])
        reuse = self.store.reusable_embeddings(doc_id) if row["embed_id"] == self.embed_id else {}
        if file_sha256(path) == row["sha256"]:      # same content: rebuild in place
            self.store.delete_document(doc_id)
            return self.ingest_file(path, reuse)
        result = self.ingest_file(path, reuse)
        if result.status in ("ingested", "needs_ocr", "duplicate"):   # duplicate: now identical to another doc
            self.store.delete_document(doc_id)
        if result.status == "ingested":
            self.overrides.copy(row["sha256"], self.store.sha_of(result.doc_id))   # keep human corrections
        return result

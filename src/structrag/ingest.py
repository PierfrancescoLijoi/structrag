"""Ingestion pipeline: parse -> infer structure (heuristics, then agent) -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

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


class Ingestor:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder,
                 llm: ChatModel | None = None):
        self.s, self.store, self.embedder, self.llm = settings, store, embedder, llm
        self.profiles = ProfileStore(settings.profiles_dir)
        self.overrides = OverrideStore(settings.profiles_dir)
        self.pii_detector: Detector | None = load_detector(settings) if settings.pii_mode == "mask" else None

    def ingest_file(self, path: Path) -> IngestResult:
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
        return self._build(path, sha, doc)

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
                "status": status, "warnings": list(doc.warnings), "pii": pii or {}}

    def _build(self, path: Path, sha: str, doc: ParsedDoc) -> IngestResult:
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
        doc_vec = self.embedder.embed([f"{doc.title}\n{doc_summary}"])[0]
        sec_vecs = self.embedder.embed([s["card"] for s in sections])
        flat = [c for s in sections for c in s["chunks"]]
        chunk_vecs = self.embedder.embed([c.ctx for c in flat]) if flat else []
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
        """Rebuild a document from its stored file, e.g. after a human correction."""
        row = self.store.conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row or not Path(row["path"]).exists():
            return IngestResult("failed", error="source file is no longer available")
        path = Path(row["path"])
        self.store.delete_document(doc_id)
        return self.ingest_file(path)

"""Hierarchical hybrid retrieval: document card -> {dense chunks, BM25 chunks, section cards} -> RRF.

Small sections are returned whole (parent-child expansion) so the model sees complete context.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .llm.client import Embedder
from .store import Store

RRF_K = 60
CANDIDATES = 40
TOP_SECTIONS = 10
SECTION_WEIGHT = 0.5
EXPAND_WORDS = 350   # sections up to this size are passed to the model whole


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: int
    doc_title: str
    section_id: int
    section_path: str
    loc: str
    text: str
    kind: str
    score: float


@dataclass(frozen=True)
class Context:
    """One numbered source block shown to the model and cited in the answer."""
    n: int
    doc_id: int
    doc_title: str
    section_path: str
    loc: str
    text: str


def _top(scores: np.ndarray, ids: list[int], n: int) -> list[int]:
    order = np.argsort(-scores)[:n]
    return [ids[i] for i in order]


class Retriever:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder):
        self.s, self.store, self.embedder = settings, store, embedder

    def _scope(self, qv: np.ndarray, query: str, doc_ids: list[int] | None) -> tuple[int, ...] | None:
        """Stage 1: shortlist documents by their summary card (skipped for small corpora)."""
        docs = [d["id"] for d in self.store.list_documents() if d["status"] != "needs_ocr"]
        if doc_ids:
            docs = [d for d in docs if d in set(doc_ids)]
        if len(docs) < self.s.doc_prefilter_min_docs:
            return tuple(sorted(docs))
        m = self.store.matrix("documents", tuple(sorted(docs)))
        picked = set(_top(m.vectors @ qv, m.ids, self.s.top_docs))
        # exact-term matches must survive even when the doc card does not mention them
        lexical = self.store.fts(query, docs, CANDIDATES)
        picked |= {c["doc_id"] for c in self.store.chunks_by_ids(lexical).values()}
        return tuple(sorted(picked))

    def search(self, query: str, doc_ids: list[int] | None = None, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_chunks
        qv = self.embedder.embed([query])[0]
        scope = self._scope(qv, query, doc_ids)
        if not scope:
            return []
        scoped = list(scope)
        rankings: list[tuple[float, list[int]]] = []

        chunks = self.store.matrix("chunks", scope)
        if chunks.ids:
            rankings.append((1.0, _top(chunks.vectors @ qv, chunks.ids, CANDIDATES)))
        rankings.append((1.0, self.store.fts(query, scoped, CANDIDATES)))

        sections = self.store.matrix("sections", scope)
        if sections.ids:
            top_sections = _top(sections.vectors @ qv, sections.ids, TOP_SECTIONS)
            by_section = self.store.section_chunk_ids(top_sections)
            rankings.append((SECTION_WEIGHT, [c for s in top_sections for c in by_section.get(s, [])]))

        fused: dict[int, float] = {}
        for weight, ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + weight / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        rows = self.store.chunks_by_ids(best)
        return [Hit(cid, r["doc_id"], r["doc_title"], r["section_id"], r["section_path"], r["loc"],
                    r["text"], r["kind"], fused[cid]) for cid in best if (r := rows.get(cid))]

    def build_context(self, hits: list[Hit]) -> list[Context]:
        """Merge hits per section; expand small sections to their full text."""
        order: dict[int, list[Hit]] = {}
        for h in hits:
            order.setdefault(h.section_id, []).append(h)
        out = []
        for n, (section_id, group) in enumerate(order.items(), 1):
            full = self.store.section_text(section_id)
            text = full if len(full.split()) <= EXPAND_WORDS else "\n\n".join(h.text for h in group)
            first = group[0]
            out.append(Context(n, first.doc_id, first.doc_title, first.section_path, first.loc, text))
        return out

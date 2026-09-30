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
MIN_CONTEXT_CHARS = 400   # do not bother adding a source with less room than this
LINK_SHARE = 4            # graph hits may claim up to 1/4 of the context budget


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
    ref: str = ""
    via: str = ""            # graph hits only: how it was reached, "{n}" = number of the citing source
    seed: int = 0            # graph hits only: chunk id of the hit that led here


@dataclass(frozen=True)
class Context:
    """One numbered source block shown to the model and cited in the answer."""
    n: int
    doc_id: int
    doc_title: str
    section_path: str
    loc: str
    text: str
    kind: str = "text"
    ref: str = ""            # thumbnail id when the best hit of this source is a figure
    score: float = 0.0       # best relevance score among the merged hits
    figures: tuple[str, ...] = ()   # thumbnail ids of the figures of this source (first one == ref)
    via: str = ""            # set when the source was pulled in by a reference, e.g. 'cited as "Figure 3" in [1]'
    chunks: tuple[int, ...] = ()   # chunk ids behind this source: where its references are looked up


def _top(scores: np.ndarray, ids: list[int], n: int) -> list[int]:
    order = np.argsort(-scores)[:n]
    return [ids[i] for i in order]


def _fuse(rankings: list[tuple[float, list[int]]]) -> dict[int, float]:
    """Weighted reciprocal-rank fusion."""
    fused: dict[int, float] = {}
    for weight, ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + weight / (RRF_K + rank + 1)
    return fused


class Retriever:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder, reranker=None):
        self.s, self.store, self.embedder, self.reranker = settings, store, embedder, reranker

    def _embed_query(self, query: str) -> np.ndarray:
        embed = getattr(self.embedder, "embed_query", self.embedder.embed)   # asymmetric models need a query mode
        return embed([query])[0]

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
        qv = self._embed_query(query)
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

        fused = _fuse(rankings)
        pool = sorted(fused, key=fused.get, reverse=True)[:max(k, self.s.rerank_candidates if self.reranker else k)]
        rows = self.store.chunks_by_ids(pool)
        pool = [cid for cid in pool if cid in rows]
        if self.reranker and pool:
            scores = self.reranker.scores(query, [rows[cid]["ctx"] for cid in pool])
            fused = dict(zip(pool, scores))
            pool.sort(key=fused.get, reverse=True)
        seen: set[str] = set()
        hits = []
        for cid in pool:
            key = " ".join(rows[cid]["text"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            r = rows[cid]
            hits.append(Hit(cid, r["doc_id"], r["doc_title"], r["section_id"], r["section_path"], r["loc"],
                            r["text"], r["kind"], fused[cid], r["ref"] or ""))
            if len(hits) == k:
                break
        return hits

    # ---- reference graph -----------------------------------------------------------------------
    def _best_in_doc(self, qv: np.ndarray, query: str, doc_id: int, n: int = 3) -> list[int]:
        """Best chunks of one cited document for the question (dense + BM25, fused, not yet reranked)."""
        m = self.store.matrix("chunks", (doc_id,))
        rankings = [(1.0, self.store.fts(query, [doc_id], CANDIDATES))]
        if m.ids:
            rankings.append((1.0, _top(m.vectors @ qv, m.ids, CANDIDATES)))
        fused = _fuse(rankings)
        return sorted(fused, key=fused.get, reverse=True)[:n]

    def follow_links(self, query: str, hits: list[Hit]) -> list[Hit]:
        """One hop from the best hits: passages they cite (figure, table, section, another document) or that
        cite them. Never chained. Only kept when the reranker finds them relevant to THIS question."""
        if not (self.s.graph and hits):
            return []
        seeds = {h.chunk_id: h for h in hits[:self.s.link_seeds]}
        direct: dict[int, tuple[int, str]] = {}      # chunk -> (seed chunk, via)
        cited_docs: dict[int, tuple[int, str]] = {}  # document -> (seed chunk, via)
        for l in self.store.links_out(list(seeds)):
            via = f'cited as "{l["label"]}" in [{{n}}]'
            if l["dst_chunk"]:
                direct.setdefault(l["dst_chunk"], (l["src"], via))
            else:
                cited_docs.setdefault(l["dst_doc"], (l["src"], via))
        for l in self.store.links_in(list(seeds)):
            direct.setdefault(l["src"], (l["dst_chunk"], f'cites [{{n}}] as "{l["label"]}"'))
        if not direct and not cited_docs:
            return []
        have = {h.chunk_id for h in hits}
        candidates = {c: v for c, v in direct.items() if c not in have}
        if cited_docs and self.reranker:                 # a document is a whole: pick its passage for the question
            qv = self._embed_query(query)
            for doc_id, (seed, via) in list(cited_docs.items())[:self.s.link_max]:
                for cid in self._best_in_doc(qv, query, doc_id):
                    if cid not in have:
                        candidates.setdefault(cid, (seed, via))
        rows = self.store.chunks_by_ids(list(candidates))
        pool = [c for c in candidates if c in rows]
        if self.reranker and pool:
            scores = dict(zip(pool, self.reranker.scores(query, [rows[c]["ctx"] for c in pool])))
            pool = sorted((c for c in pool if scores[c] >= self.s.link_min_relevance), key=scores.get, reverse=True)
        else:
            scores = dict.fromkeys(pool, 0.0)
        out, seen = [], set()
        for cid in pool[:self.s.link_max * 2]:
            r = rows[cid]
            key = " ".join(r["text"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            seed, via = candidates[cid]
            out.append(Hit(cid, r["doc_id"], r["doc_title"], r["section_id"], r["section_path"], r["loc"],
                           r["text"], r["kind"], scores[cid], r["ref"] or "", via, seed))
            if len(out) == self.s.link_max:
                break
        return out

    # ---- context -------------------------------------------------------------------------------
    def build_context(self, hits: list[Hit], max_chars: int | None = None,
                      linked: list[Hit] | tuple = ()) -> list[Context]:
        """Merge hits per section (best-ranked first); expand small sections to their full text.
        `linked` (see follow_links) are appended after the hits, whatever budget is left. Stops at `max_chars`
        so the prompt always fits the model window."""
        budget = max_chars or self.s.context_max_chars
        reserve = min(sum(len(h.text) for h in linked), budget // LINK_SHARE)   # graph hits must not be starved
        budget -= reserve
        order: dict[int, list[Hit]] = {}
        for h in hits:
            order.setdefault(h.section_id, []).append(h)
        section_figures = self.store.section_figures(list(order))
        section_chunks = self.store.section_chunk_ids(list(order))
        out: list[Context] = []
        for section_id, group in order.items():
            full = self.store.section_text(section_id)
            whole = len(full.split()) <= EXPAND_WORDS and len(full) <= budget
            text = full if whole else "\n\n".join(h.text for h in group)
            if len(text) > budget:
                if out and budget < MIN_CONTEXT_CHARS:
                    break
                text = text[:budget]
            first = group[0]
            figure = next((h for h in group if h.kind == "image"), None)
            refs = ([figure.ref] if figure else []) + (section_figures.get(section_id, []) if whole else [])
            figures = tuple(dict.fromkeys(refs))
            out.append(Context(len(out) + 1, first.doc_id, first.doc_title, first.section_path, first.loc, text,
                               "image" if figure else first.kind, figures[0] if figures else "",
                               max(h.score for h in group), figures, "",
                               tuple(section_chunks.get(section_id, [])) if whole else tuple(h.chunk_id for h in group)))
            budget -= len(text)
            if budget <= 0:
                break
        return out + self._linked_context(out, hits, linked, budget + reserve)

    def _linked_context(self, main: list[Context], hits: list[Hit], linked, budget: int) -> list[Context]:
        by_section = {(c.doc_id, c.section_path): c for c in main}
        section_of = {h.chunk_id: (h.doc_id, h.section_path) for h in hits}
        out: list[Context] = []
        for h in linked:
            citing = by_section.get(section_of.get(h.seed))
            shown = by_section.get((h.doc_id, h.section_path))
            if citing is None or budget < min(MIN_CONTEXT_CHARS, len(h.text)) or (shown and h.text in shown.text):
                continue
            text = h.text[:budget]
            figures = (h.ref,) if h.ref else ()
            out.append(Context(len(main) + len(out) + 1, h.doc_id, h.doc_title, h.section_path, h.loc, text,
                               h.kind, h.ref, h.score, figures, h.via.replace("{n}", str(citing.n)), (h.chunk_id,)))
            budget -= len(text)
        return out

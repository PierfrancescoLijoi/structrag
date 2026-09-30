"""Chat service: grounded answers with citations, or an honest "not in the documents".

Flow per question
  1. retrieve + rerank; if even the best passage is not relevant enough -> refuse without calling the model
  2. generate an answer that must cite numbered passages, or reply NO_ANSWER
  3. check every sentence against the passages it cites (grounding.py) and drop what they do not support
  4. ask the model to verify the surviving answer against the cited passages only (one logprob forward pass)
  5. answer with per-sentence citations and quotes, or refuse. Nothing unverified reaches the user.
Session memory (rolling summary + recent turns) is unchanged.
"""
from __future__ import annotations

import logging
from typing import Iterator

from . import grounding
from .config import Settings
from .llm.client import ChatModel, LLMError
from .retrieve import Context, Retriever
from .store import Store

log = logging.getLogger(__name__)
RECENT_MESSAGES = 6          # verbatim turns kept in the prompt
MEMORY_TRIGGER_CHARS = 6000  # older history beyond this is folded into the session summary
TITLE_WORDS = 6
ANSWER_MAX_TOKENS = 500
SOURCE_PREVIEW_CHARS = 600

SYSTEM = (
    "You answer questions using ONLY the numbered context passages below.\n"
    "Rules:\n"
    "1. End every sentence of your answer with the citation [n] of the passage that states it.\n"
    "2. Use only facts written in the passages. Never use outside knowledge, never guess, never infer "
    "beyond what is written, and never round or change numbers.\n"
    "3. If the passages do not contain the answer, reply with exactly: NO_ANSWER\n"
    "4. If they contain only part of the answer, give that part and say which part the documents do not cover.\n"
    "5. Be brief and answer in the language of the question.")

VERIFY = (
    "Passages:\n{passages}\n\nProposed answer:\n{answer}\n\n"
    "Is EVERY claim in the proposed answer explicitly stated in the passages above (nothing added from outside "
    "knowledge, no changed numbers)?\nA) Yes, fully supported\nB) No, at least one claim is not supported\n"
    "Reply with A or B.")


def _format_context(contexts: list[Context]) -> str:
    return "\n\n".join(f"[{c.n}] {c.doc_title} > {c.section_path} ({c.loc})\n{c.text}" for c in contexts)


def _source(c: Context, cited: bool = False, evidence: str = "") -> dict:
    return {"n": c.n, "doc_id": c.doc_id, "doc": c.doc_title, "section": c.section_path, "loc": c.loc,
            "text": c.text[:SOURCE_PREVIEW_CHARS], "kind": c.kind, "image": c.ref, "cited": cited,
            "evidence": evidence}


class ChatService:
    def __init__(self, settings: Settings, store: Store, retriever: Retriever, llm: ChatModel):
        self.s, self.store, self.retriever, self.llm = settings, store, retriever, llm

    # ---- memory ----------------------------------------------------------------------------
    def _standalone_query(self, message: str, history: list[dict]) -> str:
        """Rewrite a follow-up ('and the second one?') into a self-contained search query."""
        if not history:
            return message
        transcript = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        prompt = (f"Conversation:\n{transcript}\n\nRewrite the last user message as a standalone "
                  f"search query. Output only the query.\nLast user message: {message}")
        try:
            return self.llm.chat([{"role": "user", "content": prompt}], max_tokens=60, temperature=0) or message
        except LLMError:
            return f"{history[-1]['content'][:200]} {message}" if history[-1]["role"] == "user" else message

    def _fold_memory(self, session_id: int) -> None:
        """Summarise old turns into the session summary once the transcript grows too long."""
        msgs = self.store.messages(session_id)
        if sum(len(m["content"]) for m in msgs) < MEMORY_TRIGGER_CHARS or len(msgs) <= RECENT_MESSAGES:
            return
        old = msgs[:-RECENT_MESSAGES]
        session = self.store.get_session(session_id) or {}
        text = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in old)
        prompt = (f"Previous summary: {session.get('summary') or '(none)'}\n\nNew turns:\n{text}\n\n"
                  "Update the summary in under 150 words, keeping facts, names and open questions.")
        try:
            self.store.set_session(session_id, summary=self.llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=250, temperature=0.1))
        except LLMError:
            log.warning("memory summarisation failed; keeping previous summary")

    # ---- grounding -------------------------------------------------------------------------
    def _relevant(self, contexts: list[Context]) -> tuple[bool, float]:
        """Is the best passage relevant enough to even try? Only meaningful with a reranker (calibrated scores)."""
        best = max((c.score for c in contexts), default=float("-inf"))
        if not contexts:
            return False, best
        if self.retriever.reranker is None:
            return True, best
        return best >= self.s.min_relevance, best

    def _verified(self, answer: str, passages: list[Context]) -> tuple[bool, float]:
        """LLM check of the assembled answer against ONLY the passages it cites."""
        listing = "\n\n".join(f"[{c.n}] {c.text[:1500]}" for c in passages)
        try:
            probs = self.llm.choose([{"role": "user", "content": VERIFY.format(passages=listing, answer=answer)}],
                                    ["A", "B"])
        except LLMError as exc:
            log.warning("verifier unavailable, refusing to answer unverified: %s", exc)
            return False, 0.0
        return probs["A"] >= self.s.verifier_min, probs["A"]

    def _grounded_answer(self, question: str, contexts: list[Context], raw: str) -> tuple[grounding.Checked, dict]:
        """(checked claims, verdict). An empty `claims` list means: refuse."""
        verdict: dict = {"reason": ""}
        if grounding.is_refusal(raw):
            verdict["reason"] = "model found no answer in the passages"
            return grounding.Checked(), verdict
        checked = grounding.check_answer(raw, [c.text for c in contexts], self.s.min_support)
        verdict["dropped"] = [{"sentence": s, "why": why} for s, why in checked.dropped]
        if not checked.claims:
            verdict["reason"] = "no sentence was supported by the passages"
            return checked, verdict
        if self.s.grounding == "strict":
            cited = [c for c in contexts if c.n in checked.cited]
            ok, p = self._verified(checked.render(), cited)
            verdict["verifier"] = round(p, 3)
            if not ok:
                verdict["reason"] = "verifier rejected the answer"
                return grounding.Checked(dropped=checked.dropped), verdict
        return checked, verdict

    # ---- main entry ------------------------------------------------------------------------
    def ask(self, session_id: int, message: str, doc_ids: list[int] | None = None) -> Iterator[dict]:
        session = self.store.get_session(session_id)
        if session is None:
            yield {"type": "error", "text": "unknown session"}
            return
        history = self.store.messages(session_id)
        self.store.add_message(session_id, "user", message)
        if not history:
            self.store.set_session(session_id, title=" ".join(message.split()[:TITLE_WORDS]))

        yield {"type": "status", "text": "searching"}
        query = self._standalone_query(message, history)
        contexts = self.retriever.build_context(self.retriever.search(query, doc_ids))
        yield {"type": "sources", "query": query, "sources": [_source(c) for c in contexts]}

        relevant, best = self._relevant(contexts)
        verdict: dict = {"relevance": None if best == float("-inf") else round(best, 2), "reason": ""}
        claims: list[grounding.Claim] = []
        if not relevant and self.s.grounding != "off":
            verdict["reason"] = "no relevant passage found"
        else:
            yield {"type": "status", "text": "answering"}
            system = SYSTEM + (f"\n\nConversation memory: {session['summary']}" if session["summary"] else "")
            prompt = [{"role": "system", "content": system + "\n\nContext:\n" + (_format_context(contexts) or "(empty)")},
                      *({"role": m["role"], "content": m["content"]} for m in history[-RECENT_MESSAGES:]),
                      {"role": "user", "content": message}]
            try:
                raw = self.llm.chat(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0)
            except LLMError as exc:
                yield {"type": "error", "text": str(exc)}
                yield {"type": "done"}
                return
            if self.s.grounding == "off":
                claims = [grounding.Claim(raw, tuple(c.n for c in contexts))] if raw else []
            else:
                yield {"type": "status", "text": "verifying"}
                checked, more = self._grounded_answer(message, contexts, raw)
                claims, verdict = checked.claims, {**verdict, **more}

        by_n = {c.n: c for c in contexts}
        if claims:
            text = "".join(c.text + "".join(f"[{n}]" for n in c.cites) + " " for c in claims).strip() \
                if self.s.grounding == "off" else grounding.Checked(claims=claims).render()
            evidence = {n: next((c.evidence for c in claims if c.cites and c.cites[0] == n), "") for n in by_n}
            cited = sorted({n for c in claims for n in c.cites})
            sources = [_source(by_n[n], True, evidence.get(n, "")) for n in cited]
        else:
            text, cited, sources = grounding.refusal(message), [], []
        yield {"type": "final", "answered": bool(claims), "text": text, "citations": sources,
               "claims": [{"text": c.text, "cites": list(c.cites), "evidence": c.evidence} for c in claims],
               "verdict": verdict}
        yield {"type": "token", "text": text}
        self.store.add_message(session_id, "assistant", text, sources)
        self._fold_memory(session_id)
        yield {"type": "done"}

"""Chat service: retrieval-grounded answers with session memory (rolling summary + recent turns)."""
from __future__ import annotations

import logging
from typing import Iterator

from .config import Settings
from .llm.client import ChatModel, LLMError
from .retrieve import Context, Retriever
from .store import Store

log = logging.getLogger(__name__)
RECENT_MESSAGES = 6          # verbatim turns kept in the prompt
MEMORY_TRIGGER_CHARS = 6000  # older history beyond this is folded into the session summary
TITLE_WORDS = 6

SYSTEM = ("You answer questions about the user's local documents. Use ONLY the numbered context "
          "below. Cite sources inline like [1] or [2]. If the context does not contain the answer, "
          "say so plainly instead of guessing. Answer in the language of the question.")


def _format_context(contexts: list[Context]) -> str:
    return "\n\n".join(f"[{c.n}] {c.doc_title} > {c.section_path} ({c.loc})\n{c.text}" for c in contexts)


class ChatService:
    def __init__(self, settings: Settings, store: Store, retriever: Retriever, llm: ChatModel):
        self.s, self.store, self.retriever, self.llm = settings, store, retriever, llm

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

    def ask(self, session_id: int, message: str, doc_ids: list[int] | None = None) -> Iterator[dict]:
        session = self.store.get_session(session_id)
        if session is None:
            yield {"type": "error", "text": "unknown session"}
            return
        history = self.store.messages(session_id)
        self.store.add_message(session_id, "user", message)
        if not history:
            self.store.set_session(session_id, title=" ".join(message.split()[:TITLE_WORDS]))

        query = self._standalone_query(message, history)
        contexts = self.retriever.build_context(self.retriever.search(query, doc_ids))
        sources = [{"n": c.n, "doc_id": c.doc_id, "doc": c.doc_title, "section": c.section_path,
                    "loc": c.loc, "text": c.text[:600]} for c in contexts]
        yield {"type": "sources", "query": query, "sources": sources}

        system = SYSTEM + (f"\n\nConversation memory: {session['summary']}" if session["summary"] else "")
        prompt = [{"role": "system", "content": system + "\n\nContext:\n" + (_format_context(contexts) or "(empty)")},
                  *({"role": m["role"], "content": m["content"]} for m in history[-RECENT_MESSAGES:]),
                  {"role": "user", "content": message}]
        answer: list[str] = []
        try:
            for token in self.llm.stream(prompt):
                answer.append(token)
                yield {"type": "token", "text": token}
        except LLMError as exc:
            yield {"type": "error", "text": str(exc)}
        text = "".join(answer)
        if text:
            self.store.add_message(session_id, "assistant", text, sources)
            self._fold_memory(session_id)
        yield {"type": "done"}

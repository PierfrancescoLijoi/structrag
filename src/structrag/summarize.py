"""Two-level index summaries: document card and per-section cards.

Default `lead` mode needs no model: the summary is the introductory paragraph plus the outline.
`llm` mode asks the local model for a short summary, only for sections/documents above the
size thresholds; any model failure silently falls back to the lead summary.
"""
from __future__ import annotations

from .config import Settings
from .ir import Section
from .llm.client import ChatModel, LLMError

LEAD_WORDS = 60
DOC_LEAD_WORDS = 120
LLM_INPUT_WORDS = 1200
OUTLINE_MAX = 25


def lead(text: str, words: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:words]) + (" ..." if len(tokens) > words else "")


def _own_text(section: Section) -> str:
    return " ".join(b.text for b in section.blocks)


def _llm_summary(llm: ChatModel, title: str, text: str) -> str | None:
    prompt = (f"Summarize this section titled '{title}' in at most 2 sentences, "
              f"keeping names, numbers and key terms:\n\n{lead(text, LLM_INPUT_WORDS)}")
    try:
        return llm.chat([{"role": "user", "content": prompt}], max_tokens=120, temperature=0.1) or None
    except LLMError:
        return None


def section_summary(section: Section, llm: ChatModel | None, settings: Settings) -> str:
    text = _own_text(section)
    if llm and settings.summary_mode == "llm" and section.words >= settings.section_summary_words:
        if (s := _llm_summary(llm, section.title, text)):
            return s
    parts = [lead(text, LEAD_WORDS)] if text else []
    if section.children:
        parts.append("Subsections: " + "; ".join(c.title for c in section.children[:10]))
    return " ".join(parts)


def outline_lines(root: Section) -> list[str]:
    lines = [f"{'  ' * (s.level - 1)}{s.title}" for s, _ in root.walk() if s.level and s.level <= 2]
    return lines[:OUTLINE_MAX]


def document_summary(root: Section, total_words: int, llm: ChatModel | None, settings: Settings) -> str:
    """Outline + introductory paragraph (or an LLM summary of both for long documents)."""
    intro = next((_own_text(s) for s, _ in root.walk() if s.blocks), "")
    outline = "; ".join(t.strip() for t in outline_lines(root))
    fallback = f"{lead(intro, DOC_LEAD_WORDS)}\nOutline: {outline}".strip()
    if llm and settings.summary_mode == "llm" and total_words >= settings.long_doc_words:
        text = f"Outline: {outline}\nIntroduction: {lead(intro, 400)}"
        if (s := _llm_summary(llm, root.title, text)):
            return s
    return fallback

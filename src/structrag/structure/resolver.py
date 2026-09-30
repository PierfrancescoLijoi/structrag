"""Agentic fallback for ambiguous structure, using logprob-based decisions (no free generation).

Two decision types, both multiple choice so a 4B local model answers reliably and with a
confidence: (1) is this candidate line a section heading? (2) which row is the table header?
Answers below `agent_accept_prob` are not applied: they go to the human review queue.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import Settings
from ..ir import TABLE, Block, ParsedDoc
from ..llm.client import ChatModel, LLMError
from .heuristics import Inference

LABELS = "ABCDEFGH"
CONTEXT_CHARS = 240
OUTLINE_ITEMS = 12
SYSTEM = ("You are a document structure analyst. Answer with exactly one capital letter "
          "and nothing else.")


@dataclass(frozen=True)
class Decision:
    kind: str                 # "heading" | "table_header"
    idx: int                  # block index
    text: str
    probs: dict[str, float]
    chosen: str
    accepted: bool
    style_key: str = ""
    level: int = 0            # heading level (0 = body) for kind == "heading"
    options: dict[str, str] | None = None


def _clip(text: str) -> str:
    return text if len(text) <= CONTEXT_CHARS else text[:CONTEXT_CHARS] + "..."


def _ask(llm: ChatModel, state: str, question: str, options: dict[str, str]) -> dict[str, float]:
    listing = "\n".join(f"{label}) {desc}" for label, desc in options.items())
    user = f"{state}\n\nQuestion: {question}\n{listing}\nAnswer:"
    return llm.choose([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                      list(options))


def _heading_state(doc: ParsedDoc, levels: list[int | None], idx: int, body_key: str) -> str:
    blocks = doc.blocks
    outline = [f"{'  ' * (lv - 1)}- {_clip(b.text)}" for b, lv in zip(blocks[:idx], levels[:idx]) if lv]
    prev = " / ".join(_clip(b.text) for b in blocks[max(0, idx - 2):idx]) or "(start of document)"
    nxt = " / ".join(_clip(b.text) for b in blocks[idx + 1:idx + 3]) or "(end of document)"
    cand = blocks[idx]
    return (f"Document: {doc.title} ({doc.format})\n"
            f"Headings found so far:\n" + ("\n".join(outline[-OUTLINE_ITEMS:]) or "(none)") + "\n"
            f"Text before candidate: {prev}\n"
            f"CANDIDATE LINE: {_clip(cand.text)}\n"
            f"Candidate style: size={cand.size} bold={cand.bold} (body style: {body_key})\n"
            f"Text after candidate: {nxt}")


def resolve_headings(doc: ParsedDoc, inf: Inference, llm: ChatModel,
                     settings: Settings) -> tuple[list[int | None], list[Decision]]:
    """Confirm or drop the ambiguous heading candidates. Returns updated levels and decisions."""
    levels = list(inf.levels)
    decisions: list[Decision] = []
    options = {"A": "a section heading (short title introducing the text that follows)",
               "B": "ordinary body text, a caption, a label or a table/list fragment"}
    for idx in inf.ambiguous[:settings.agent_budget_per_doc]:
        try:
            probs = _ask(llm, _heading_state(doc, levels, idx, inf.body_key),
                         "What is the candidate line?", options)
        except LLMError:
            break   # model server down: keep heuristics, nothing is lost
        chosen = max(probs, key=probs.get)
        accepted = probs[chosen] >= settings.agent_accept_prob
        provisional = levels[idx] or 1
        if accepted and chosen == "B":
            levels[idx] = None
        elif not accepted:
            levels[idx] = None   # unsure => do not split the document on a doubtful title
        decisions.append(Decision("heading", idx, doc.blocks[idx].text, probs, chosen, accepted,
                                  doc.blocks[idx].style_key, provisional if chosen == "A" else 0, options))
    return levels, decisions


def _row_desc(row: tuple[str, ...]) -> str:
    return _clip(" | ".join(c if c else "(empty)" for c in row))


def resolve_table_headers(doc: ParsedDoc, llm: ChatModel,
                          settings: Settings) -> tuple[ParsedDoc, list[Decision]]:
    """Ask which of the first rows is the header for tables flagged ambiguous by the parser."""
    blocks = list(doc.blocks)
    decisions: list[Decision] = []
    for idx, b in enumerate(blocks):
        if b.kind != TABLE or not b.meta.get("header_ambiguous"):
            continue
        rows = b.rows[:len(LABELS)][:5]
        options = {LABELS[i]: f"row {i + 1}: {_row_desc(r)}" for i, r in enumerate(rows)}
        state = f"Spreadsheet range {b.meta.get('range')}. First rows of the table:"
        try:
            probs = _ask(llm, state, "Which row contains the column headers?", options)
        except LLMError:
            break
        chosen = max(probs, key=probs.get)
        accepted = probs[chosen] >= settings.agent_accept_prob
        if accepted:
            new_idx = LABELS.index(chosen)
            meta = {**b.meta, "header_row": new_idx, "column_names": list(b.rows[new_idx]),
                    "header_ambiguous": False}
            blocks[idx] = replace(b, meta=meta)
        decisions.append(Decision("table_header", idx, b.meta.get("range", ""), probs, chosen,
                                  accepted, options=options))
    return replace(doc, blocks=tuple(blocks)), decisions

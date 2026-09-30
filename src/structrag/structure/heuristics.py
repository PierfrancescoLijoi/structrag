"""Heading inference cascade: explicit structure -> learned profile -> numbering/typography -> shape.

Each stage is cheap and deterministic. Candidates that only weak signals support (bold-only,
ALL CAPS, single-level numbering, unknown layout) are returned as `ambiguous` for the agent.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..ir import LIST_ITEM, PARA, Block, ParsedDoc

MAX_HEADING_CHARS = 120
MAX_HEADING_SHARE = 0.60       # more candidates than this share of blocks => layout is suspect
MAX_WEAK_SHARE = 0.60          # bold-only candidates are dropped beyond this share
MIN_BLOCKS_FOR_SHARE = 20      # tiny documents legitimately have a high title:body ratio
LARGER_THAN_BODY = 1.10        # size ratio that makes a heading "strong"
MAX_LEVELS = 4
MIN_NUMBERED_HITS = 3
LONG_DOC_WORDS = 800           # docs longer than this with no headings need the agent
SHAPE_MAX_WORDS = 10

NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,4})[.)]?\s+(?P<rest>[A-ZÀ-Ý0-9\"'(].*)$")
KEYWORD = re.compile(
    r"^(chapter|capitolo|section|sezione|part|parte|appendix|appendice|allegato|articolo|art\.)\s+"
    r"([0-9]+|[ivxlc]+|[a-z])\b", re.I)
TRAILING = tuple(".,;")
MIN_HEADING_LETTERS = 3
MIN_LETTER_SHARE = 0.5         # table rows ("73.7 79.4 84.6") and figure debris are not headings


@dataclass(frozen=True)
class Inference:
    levels: list[int | None]
    confidence: float
    strategy: str
    ambiguous: list[int] = field(default_factory=list)
    body_key: str = ""
    notes: tuple[str, ...] = ()


def dominant_style(blocks: tuple[Block, ...]) -> str:
    """Style key covering the most characters of body paragraphs."""
    weight: Counter[str] = Counter()
    for b in blocks:
        if b.kind in (PARA, LIST_ITEM) and b.level_hint is None:
            weight[b.style_key] += len(b.text)
    return weight.most_common(1)[0][0] if weight else ""


def _wordlike(text: str) -> bool:
    compact = [c for c in text if not c.isspace()]
    letters = sum(c.isalpha() for c in compact)
    return letters >= MIN_HEADING_LETTERS and letters / len(compact) >= MIN_LETTER_SHARE


def _headingish(b: Block) -> bool:
    return (b.kind in (PARA, LIST_ITEM) and 0 < len(b.text) <= MAX_HEADING_CHARS
            and not b.text.rstrip().endswith(TRAILING) and b.level_hint is None and _wordlike(b.text))


def _body_size(blocks: tuple[Block, ...]) -> float | None:
    weight: Counter[float] = Counter()
    for b in blocks:
        if b.kind == PARA and b.size:
            weight[round(b.size * 2) / 2] += len(b.text)
    return weight.most_common(1)[0][0] if weight else None


def _explicit(doc: ParsedDoc) -> list[int | None] | None:
    hinted = [b.level_hint for b in doc.blocks]
    count = sum(h is not None for h in hinted)
    words = sum(len(b.text.split()) for b in doc.blocks)
    if doc.format in ("pptx", "xlsx") and count:
        return hinted
    if count >= 2 or (count == 1 and words < LONG_DOC_WORDS):
        return hinted
    return None


def _numbering(blocks: tuple[Block, ...]) -> dict[int, tuple[int, bool]]:
    """idx -> (level, strong). Multi-level numbers are strong; single-level are runs-checked."""
    hits: dict[int, tuple[int, bool]] = {}
    for i, b in enumerate(blocks):
        if not _headingish(b):
            continue
        if (m := NUMBERED.match(b.text)):
            depth = m.group("num").count(".") + 1
            hits[i] = (min(depth, MAX_LEVELS), depth > 1)
        elif KEYWORD.match(b.text):
            hits[i] = (2 if b.text.lower().startswith(("art", "articolo")) else 1, True)
    # single-level runs of >=3 consecutive numbered lines are lists, not headings
    for i in sorted(hits):
        run = [j for j in (i - 1, i, i + 1) if j in hits and not hits[j][1]]
        if not hits[i][1] and len(run) == 3:
            hits.pop(i, None)
    return hits if len(hits) >= MIN_NUMBERED_HITS else {}


def _typography(blocks: tuple[Block, ...]) -> dict[int, tuple[int, bool]]:
    body = _body_size(blocks)
    body_bold = dominant_style(blocks).split("|")[1:2] == ["1"]
    cands: dict[int, tuple[float, bool]] = {}
    for i, b in enumerate(blocks):
        if not _headingish(b):
            continue
        if body and b.size and b.size >= body * LARGER_THAN_BODY:
            cands[i] = (b.size, True)
        elif b.bold and not body_bold and (not body or not b.size or abs(b.size - body) <= 0.5):
            cands[i] = (b.size or 0.0, False)   # bold-only: size unknown or equal to body
    total = max(1, len(blocks))
    if not cands:
        return {}
    guard = len(blocks) >= MIN_BLOCKS_FOR_SHARE
    if guard and len(cands) / total > MAX_HEADING_SHARE:
        return {}
    if guard and sum(not strong for _, strong in cands.values()) / total > MAX_WEAK_SHARE:
        cands = {i: v for i, v in cands.items() if v[1]}
        if not cands:
            return {}
    ranks = sorted({(round(s), strong) for s, strong in cands.values()}, reverse=True)
    level_of = {r: min(n + 1, MAX_LEVELS) for n, r in enumerate(ranks)}
    return {i: (level_of[(round(s), strong)], strong) for i, (s, strong) in cands.items()}


def _shape(blocks: tuple[Block, ...]) -> dict[int, tuple[int, bool]]:
    """Last resort: short unpunctuated lines followed by a much longer paragraph."""
    out = {}
    for i, b in enumerate(blocks[:-1]):
        nxt = blocks[i + 1]
        if _headingish(b) and len(b.text.split()) <= SHAPE_MAX_WORDS and len(nxt.text) >= 3 * len(b.text):
            letters = [c for c in b.text if c.isalpha()]
            caps = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
            out[i] = (1, caps and len(letters) >= 4)
    return out


def infer(doc: ParsedDoc, profile: dict[str, int] | None = None) -> Inference:
    blocks = doc.blocks
    body_key = dominant_style(blocks)
    words = sum(len(b.text.split()) for b in blocks)

    hinted = _explicit(doc)
    if hinted is not None and doc.format != "pdf":
        return Inference(hinted, 0.95, "explicit", [], body_key)

    if profile:
        levels = [profile.get(b.style_key) or None if _headingish(b) else None for b in blocks]
        if any(levels):
            return Inference(levels, 0.90, "profile", [], body_key)

    cands: dict[int, tuple[int, bool]] = {}
    strategy = "none"
    numbered, typo = _numbering(blocks), _typography(blocks)
    if numbered or typo:
        cands = {**typo, **numbered}   # numbering depth wins over typography rank
        strategy = "numbering+typography" if numbered and typo else ("numbering" if numbered else "typography")
    elif words >= LONG_DOC_WORDS or doc.format == "pdf":
        cands = _shape(blocks)
        strategy = "shape" if cands else "none"

    levels: list[int | None] = [b.level_hint for b in blocks]   # keep partial explicit hints
    ambiguous: list[int] = []
    for i, (level, strong) in cands.items():
        if hinted is not None and not strong:
            continue                       # PDF bookmarks are partial: only strong extras join them
        levels[i] = level
        if not strong:
            ambiguous.append(i)
    if hinted is not None:                 # PDF outline lists top-level chapters only: add numbered sub-headings
        strategy = "outline+" + strategy if cands else "explicit"
        return Inference(levels, 0.90, strategy, ambiguous, body_key)
    if not cands:
        conf = 0.2 if words >= LONG_DOC_WORDS else 0.6
        return Inference(levels, conf, "none", [], body_key, ("no headings detected",))
    conf = 0.85 if not ambiguous else max(0.3, 0.85 - 0.5 * len(ambiguous) / len(cands))
    return Inference(levels, conf, strategy, ambiguous, body_key)

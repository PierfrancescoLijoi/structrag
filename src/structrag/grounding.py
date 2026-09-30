"""Answer grounding: every sentence of an answer must be backed by a retrieved passage, or it is removed.

Pure functions, no model calls (the LLM verifier lives in chat.py). Per sentence we
  * keep only citations that point at a real passage,
  * require the sentence's content words to be present in the passage it cites (else re-attribute it to the
    passage that does contain them, else drop it),
  * require every number / year in the sentence to appear in that passage (small models love to round or
    invent figures),
  * attach the passage sentence that best supports it, so the UI can show the quote.
If nothing survives, the caller refuses instead of answering.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .textutil import STOPWORDS, WORD

CITE = re.compile(r"\[(\d{1,2})\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9À-ÖØ-Þ])|\n+")
THOUSANDS = re.compile(r"(?<=\d)[,.\s  ](?=\d{3}(?!\d))")
DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
NO_ANSWER = "NO_ANSWER"
# words a model adds to *talk about* the sources rather than to state facts
META = frozenset("context passage passages document documents answer answers according provided mentioned stated "
                 "states state based given text information source sources paragraph section question contesto "
                 "passaggio documento documenti risposta secondo fornito indicato testo fonte fonti domanda".split())
MIN_TERM_LEN = 3


@dataclass(frozen=True)
class Claim:
    text: str                    # the sentence without citation markers
    cites: tuple[int, ...]       # 1-based passage numbers that support it
    evidence: str = ""           # best-matching sentence of the first cited passage
    support: float = 0.0         # content-word coverage in the cited passage(s)


@dataclass
class Checked:
    claims: list[Claim] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)   # (sentence, reason)

    @property
    def cited(self) -> list[int]:
        return sorted({n for c in self.claims for n in c.cites})

    def render(self) -> str:
        """'Sentence text [1][2].' : the citation goes before the final punctuation."""
        parts = []
        for c in self.claims:
            text = c.text.rstrip()
            punct = re.search(r"[.!?]+$", text)
            tail = punct.group(0) if punct else ""
            parts.append(f"{text[:len(text) - len(tail)].rstrip()} {''.join(f'[{n}]' for n in c.cites)}{tail}")
        return " ".join(parts)


def _fold(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in word if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    """Cheap EN/IT stemmer: enough to match plurals and gender endings, nothing linguistic."""
    w = _fold(word)
    if len(w) > 5 and w[-1] in "aeio":
        w = w[:-1]
    if len(w) > 4 and w.endswith("s"):
        w = w[:-1]
    return w


def content_terms(text: str) -> set[str]:
    text = CITE.sub(" ", text)
    return {_stem(w) for w in WORD.findall(text.lower())
            if len(w) >= MIN_TERM_LEN and w not in STOPWORDS and w not in META and not w.isdigit()}


def numbers(text: str) -> set[str]:
    """Numbers in a canonical form ('1,720,320' == '1720320', '1,5' == '1.5'); citation markers ignored."""
    text = DECIMAL_COMMA.sub(".", THOUSANDS.sub("", CITE.sub(" ", text)))
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in NUMBER.findall(text)}


def coverage(sentence: str, passage: str, passage_terms: set[str] | None = None) -> float:
    terms = content_terms(sentence)
    if not terms:
        return 1.0
    have = passage_terms if passage_terms is not None else content_terms(passage)
    return len(terms & have) / len(terms)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if s and s.strip()]


def best_evidence(sentence: str, passage: str) -> str:
    """The passage sentence sharing most content words with `sentence` (the quote to show next to a citation)."""
    terms = content_terms(sentence)
    candidates = split_sentences(passage) or [passage]
    scored = [(len(terms & content_terms(c)) / max(1, len(terms)), -abs(len(c) - 160), c) for c in candidates]
    return max(scored)[2][:400]


def _cited(sentence: str, n_passages: int) -> tuple[int, ...]:
    seen: list[int] = []
    for m in CITE.finditer(sentence):
        k = int(m.group(1))
        if 1 <= k <= n_passages and k not in seen:
            seen.append(k)
    return tuple(seen)


def check_answer(answer: str, passages: list[str], min_support: float) -> Checked:
    """Keep the sentences the passages back up. Passage numbers are 1-based, as in the prompt."""
    pterms = [content_terms(p) for p in passages]
    pnums = [numbers(p) for p in passages]
    out = Checked()
    for raw in split_sentences(answer):
        body = re.sub(r"\s+([.,;:!?])", r"\1", CITE.sub("", raw)).strip(" \t-•*")
        if len(re.sub(r"\W", "", body)) < 2:
            continue
        nums = numbers(body)
        declared = _cited(raw, len(passages))
        order = list(declared) + [k for k in range(1, len(passages) + 1) if k not in declared]   # cited first

        def fits(k: int) -> bool:
            return coverage(body, passages[k - 1], pterms[k - 1]) >= min_support and nums <= pnums[k - 1]

        chosen = next((k for k in order if fits(k)), None)
        if chosen is None:
            best_cov = max((coverage(body, passages[k - 1], pterms[k - 1]) for k in order), default=0.0)
            reason = "number not in the sources" if nums and best_cov >= min_support else "not supported by the sources"
            out.dropped.append((body, reason))
            continue
        cites = tuple(k for k in declared if fits(k)) or (chosen,)
        out.claims.append(Claim(body, cites, best_evidence(body, passages[cites[0] - 1]),
                                round(coverage(body, passages[cites[0] - 1], pterms[cites[0] - 1]), 2)))
    return out


ITALIAN_MARKERS = frozenset("il lo la gli le un una di del della che come quando dove quanto quale quali chi perché "
                            "non sono è era ha hanno fu nel nella dei delle con per".split())
REFUSALS = {
    "it": "Non ho trovato nei documenti abbastanza informazioni per rispondere a questa domanda.",
    "en": "I could not find enough information in the documents to answer this question.",
}


def language_of(text: str) -> str:
    words = WORD.findall(text.lower())
    italian = sum(w in ITALIAN_MARKERS for w in words)
    return "it" if italian and italian >= max(1, len(words) // 8) else "en"


def refusal(question: str) -> str:
    return REFUSALS[language_of(question)]


def is_refusal(text: str) -> bool:
    return NO_ANSWER in text or text.strip() in REFUSALS.values()

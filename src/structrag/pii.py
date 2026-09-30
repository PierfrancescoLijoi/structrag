"""PII masking with rizzo-pii (MIT, local, CPU): https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B

Runs before embedding and storage, so masked values never reach the index. The same value gets the
same placeholder within one document ("[FULLNAME_1]"), which keeps retrieval on names coherent.
Needs the optional extra: `pip install structrag[pii]` (transformers + torch, model downloads once).
"""
from __future__ import annotations

from typing import Callable

from .config import Settings

# Identifiers only. DATE, AMOUNT, ORG, AGE, GENDER, CITY... are left readable: masking them wrecks RAG.
MASK_LABELS = frozenset({
    "FULLNAME", "EMAIL", "TELEPHONENUM", "STREET", "BUILDINGNUM", "ZIPCODE", "IBAN",
    "CREDITCARDNUMBER", "CF", "PIVA", "CATASTO", "DOCID", "ID_DOC", "TARGA"})

Detector = Callable[[str], list[dict]]   # text -> [{"entity_group", "start", "end", "score"}]


def load_detector(settings: Settings) -> Detector:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("PII masking needs the extra: pip install 'structrag[pii]'") from exc
    nlp = pipeline("token-classification", model=settings.pii_model, aggregation_strategy="simple")
    return nlp


class PiiMasker:
    """One instance per document: placeholder numbering is stable across all its texts."""

    def __init__(self, detect: Detector, min_score: float):
        self.detect, self.min_score = detect, min_score
        self._ids: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def _placeholder(self, label: str, value: str) -> str:
        key = (label, value.casefold())
        if key not in self._ids:
            self._counts[label] = self._counts.get(label, 0) + 1
            self._ids[key] = f"[{label}_{self._counts[label]}]"
        return self._ids[key]

    def mask(self, text: str) -> str:
        if not text.strip():
            return text
        spans = sorted((e for e in self.detect(text)
                        if e["entity_group"] in MASK_LABELS and e["score"] >= self.min_score),
                       key=lambda e: e["start"])
        out, pos = [], 0
        for e in spans:
            if e["start"] < pos:     # overlapping detection: keep the first
                continue
            out += [text[pos:e["start"]], self._placeholder(e["entity_group"], text[e["start"]:e["end"]])]
            pos = e["end"]
        return "".join(out) + text[pos:]

    @property
    def counts(self) -> dict[str, int]:
        """Distinct masked values per label, e.g. {"FULLNAME": 2, "CF": 1}."""
        return dict(self._counts)

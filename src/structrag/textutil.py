"""Tokenisation shared by the offline embedder and the FTS query builder."""
from __future__ import annotations

import re

WORD = re.compile(r"\w{2,}", re.UNICODE)
# Minimal English + Italian stopwords: they add noise to lexical scoring and carry no topic.
STOPWORDS = frozenset("""
the a an and or of to in on at for with by is are was were be been it its this that these those as from
how what who whom which when where why do does did not no yes can could should would will
il lo la i gli le un uno una e ed o di da del della dei delle degli al allo alla ai alle
in con su per tra fra che chi cui come dove quando perché non si sono è era ho ha hanno
""".split())


def terms(text: str) -> list[str]:
    """Lowercased content words (stopwords removed unless that would leave nothing)."""
    words = WORD.findall(text.lower())
    return [w for w in words if w not in STOPWORDS] or words

"""Learned layout profiles: the self-feeding part of structure inference.

When the agent (or a human reviewer) classifies a typography style as heading level N or body,
the vote is stored under a profile keyed by (format, body style). After enough concordant votes
the mapping is applied deterministically next time, so the LLM is only paid for once per layout.
Profiles are plain JSON in `profiles/`, meant to be committed to the repo.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

MIN_VOTES = 2
MIN_AGREEMENT = 0.8
HUMAN_WEIGHT = 3   # a human correction outweighs a single model decision


class ProfileStore:
    def __init__(self, directory: Path):
        self.dir = directory

    def _path(self, fmt: str, body_key: str) -> Path:
        digest = hashlib.sha1(body_key.encode()).hexdigest()[:8]
        return self.dir / f"{fmt}-{digest}.json"

    def _load(self, fmt: str, body_key: str) -> dict:
        path = self._path(fmt, body_key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # corrupt profile: start over rather than block ingestion
        return {"format": fmt, "body_key": body_key, "votes": {}}

    def get(self, fmt: str, body_key: str) -> dict[str, int]:
        """style_key -> level (0 = body) for styles with enough concordant votes."""
        votes = self._load(fmt, body_key)["votes"]
        mapping = {}
        for style, counter in votes.items():
            total = sum(counter.values())
            level, top = max(counter.items(), key=lambda kv: kv[1])
            if total >= MIN_VOTES and top / total >= MIN_AGREEMENT:
                mapping[style] = int(level)
        return mapping

    def learn(self, fmt: str, body_key: str, observations: list[tuple[str, int]],
              human: bool = False) -> None:
        """observations: (style_key, level) pairs, level 0 meaning body text."""
        if not observations:
            return
        profile = self._load(fmt, body_key)
        weight = HUMAN_WEIGHT if human else 1
        for style, level in observations:
            counter = Counter(profile["votes"].get(style, {}))
            counter[str(level)] += weight
            profile["votes"][style] = dict(counter)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(fmt, body_key).write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")


class OverrideStore:
    """Per-document human corrections, keyed by file hash. Applied before any heuristic or agent."""

    def __init__(self, directory: Path):
        self.dir = directory / "overrides"

    def _path(self, sha: str) -> Path:
        return self.dir / f"{sha[:16]}.json"

    def get(self, sha: str) -> dict:
        path = self._path(sha)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"headings": {}, "headers": {}}

    def set(self, sha: str, section: str, key: str, value: int) -> None:
        data = self.get(sha)
        data[section][key] = value
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(sha).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

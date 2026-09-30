"""Deterministic in-process model doubles for tests and offline demos."""
from __future__ import annotations

from typing import Callable, Iterator


class FakeLLM:
    """`chooser(user_text, labels) -> probs` scripts the decisions; `reply` scripts chat output."""

    def __init__(self, chooser: Callable[[str, list[str]], dict[str, float]] | None = None,
                 reply: str = "ok"):
        self.chooser = chooser
        self.reply = reply
        self.choose_calls: list[str] = []

    def chat(self, messages, max_tokens=512, temperature=0.2) -> str:
        return self.reply

    def stream(self, messages, max_tokens=1024, temperature=0.2) -> Iterator[str]:
        yield from self.reply.split(" ")

    def choose(self, messages, labels) -> dict[str, float]:
        user = messages[-1]["content"]
        self.choose_calls.append(user)
        if self.chooser:
            return self.chooser(user, labels)
        return {label: 1 / len(labels) for label in labels}

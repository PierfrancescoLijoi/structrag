"""OpenAI-compatible local model client: chat, streaming, embeddings, logprob-based choices.

All traffic goes to `Settings.base_url` (localhost by default). Nothing leaves the machine.
"""
from __future__ import annotations

import json
import math
import zlib
from typing import Iterator, Protocol

import httpx
import numpy as np

from ..config import Settings
from ..textutil import terms

CHAT_TIMEOUT = 180.0
EMBED_BATCH = 32
HASH_DIMS = 512
TOP_LOGPROBS = 20


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.2) -> Iterator[str]: ...
    def choose(self, messages: list[dict], labels: list[str]) -> dict[str, float]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Offline feature-hashing embedder (signed, sublinear tf). Lexical only: no synonyms."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), HASH_DIMS), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in terms(text):
                h = zlib.crc32(token.encode())
                out[row, h % HASH_DIMS] += 1.0 if (h >> 16) & 1 else -1.0
        return _unit(np.sign(out) * np.sqrt(np.abs(out)))


class LLM:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.s = settings
        self.http = client or httpx.Client(base_url=settings.base_url, timeout=CHAT_TIMEOUT)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self.http.post(path, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"model server error on {path}: {exc}") from exc

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2) -> str:
        data = self._post("/chat/completions", {
            "model": self.s.chat_model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature})
        return data["choices"][0]["message"]["content"].strip()

    def stream(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.2) -> Iterator[str]:
        payload = {"model": self.s.chat_model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "stream": True}
        try:
            with self.http.stream("POST", "/chat/completions", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"model server error while streaming: {exc}") from exc

    def choose(self, messages: list[dict], labels: list[str]) -> dict[str, float]:
        """Probability over `labels` from the first-token distribution (Jev / Open JEV style).

        No text is generated: the answer letter's logprobs are read directly, so the result is
        a single fast forward pass and carries a usable confidence.
        """
        data = self._post("/chat/completions", {
            "model": self.s.chat_model, "messages": messages, "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP_LOGPROBS})
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("server returned no logprobs; needs Ollama >= 0.12.11 or llama-server") from exc
        mass = {label: 0.0 for label in labels}
        for entry in top:
            token = entry["token"].strip()
            if token in mass:
                mass[token] += math.exp(entry["logprob"])
        total = sum(mass.values())
        if total == 0:
            return {label: 1 / len(labels) for label in labels}
        return {label: p / total for label, p in mass.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            data = self._post("/embeddings", {"model": self.s.embed_model, "input": texts[i:i + EMBED_BATCH]})
            vectors.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        return _unit(np.asarray(vectors, dtype=np.float32))

    def health(self) -> dict:
        try:
            r = self.http.get("/models", timeout=3.0)
            names = [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, ValueError):
            return {"ok": False, "models": []}
        # llama-server serves whatever is loaded and lists it by file name: a lone model is the chat model
        return {"ok": True, "models": names,
                "chat_ready": self.s.chat_model in names or len(names) == 1, "embed_ready": self.s.embed_model in names}


def get_embedder(settings: Settings, llm: LLM | None = None) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    if settings.embedder == "local":
        from .local import LocalEmbedder
        return LocalEmbedder(settings)
    return llm or LLM(settings)

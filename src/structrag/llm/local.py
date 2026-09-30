"""Fully local embedding + reranking on CPU via ONNX (fastembed). No server, no GPU, no API key.

Models are open source and downloaded once to the fastembed cache. Threads are capped so the
machine stays usable while ingesting.
"""
from __future__ import annotations

import os

import numpy as np

from ..config import Settings

MAX_THREADS = 6
EMBED_BATCH = 16
RERANK_BATCH = 8

# Models fastembed does not ship: (dim, onnx file in the HF repo, query prefix, passage prefix).
# E5 models are trained with these prefixes and lose accuracy without them.
CUSTOM_MODELS = {
    "intfloat/multilingual-e5-small": (384, "onnx/model.onnx", "query: ", "passage: "),
    "intfloat/multilingual-e5-base": (768, "onnx/model.onnx", "query: ", "passage: "),
}


# Cross-encoders fastembed does not ship: (onnx file in the HF repo, size in GB). All Apache-2.0 / MIT.
CUSTOM_RERANKERS = {
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": ("onnx/model.onnx", 0.47),   # multilingual (incl. Italian)
}


def _threads(settings: Settings) -> int:
    return settings.local_threads or max(1, min(MAX_THREADS, (os.cpu_count() or 4) // 2))


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class LocalEmbedder:
    def __init__(self, settings: Settings):
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as exc:
            raise RuntimeError("local embeddings need: pip install 'structrag[local]'") from exc
        name = settings.local_embed_model
        self.query_prefix = self.passage_prefix = ""
        if name in CUSTOM_MODELS:
            dim, model_file, self.query_prefix, self.passage_prefix = CUSTOM_MODELS[name]
            if name not in {m["model"] for m in TextEmbedding.list_supported_models()}:
                TextEmbedding.add_custom_model(model=name, pooling=PoolingType.MEAN, normalization=True,
                                               sources=ModelSource(hf=name), dim=dim, model_file=model_file)
        self.model = TextEmbedding(name, threads=_threads(settings))
        self.max_chars = settings.local_max_chars

    def _run(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), np.float32)
        clipped = [prefix + t[:self.max_chars] for t in texts]
        return _unit(np.asarray(list(self.model.embed(clipped, batch_size=EMBED_BATCH)), dtype=np.float32))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Documents / passages."""
        return self._run(texts, self.passage_prefix)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self._run(texts, self.query_prefix)


class LocalReranker:
    def __init__(self, settings: Settings):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise RuntimeError("local reranking needs: pip install 'structrag[local]'") from exc
        name = settings.local_rerank_model
        if name in CUSTOM_RERANKERS and name not in {m["model"] for m in TextCrossEncoder.list_supported_models()}:
            from fastembed.common.model_description import ModelSource
            model_file, size = CUSTOM_RERANKERS[name]
            TextCrossEncoder.add_custom_model(model=name, sources=ModelSource(hf=name), model_file=model_file,
                                              size_in_gb=size)
        self.model = TextCrossEncoder(name, threads=_threads(settings))
        self.max_chars = settings.rerank_max_chars

    def scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # Batches are padded to their longest member: sorting by length removes most of the wasted compute.
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        raw = self.model.rerank(query, [passages[i][:self.max_chars] for i in order], batch_size=RERANK_BATCH)
        out = [0.0] * len(passages)
        for i, s in zip(order, raw):
            out[i] = float(s)
        return out

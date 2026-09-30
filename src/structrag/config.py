"""Runtime settings, read once from STRUCTRAG_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"STRUCTRAG_{name}", default)


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible local server works (Ollama, llama.cpp llama-server, LM Studio).
    base_url: str = field(default_factory=lambda: _env("BASE_URL", "http://127.0.0.1:11434/v1"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "qwen3-embedding:0.6b"))
    # "server" = OpenAI-compatible /embeddings; "local" = ONNX on CPU (fastembed); "hash" = offline lexical.
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "local"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    profiles_dir: Path = field(default_factory=lambda: Path(_env("PROFILES_DIR", "profiles")))

    # Structure inference: below this probability the agent's answer goes to human review.
    agent_accept_prob: float = 0.70
    agent_budget_per_doc: int = 40

    # Chunking (words; ~1.3 tokens per word).
    chunk_target_words: int = 180
    chunk_max_words: int = 260
    table_rows_per_chunk: int = 20

    # Two-level index: docs above these thresholds get a doc summary and section summaries.
    long_doc_words: int = 4000
    section_summary_words: int = 400
    summary_mode: str = field(default_factory=lambda: _env("SUMMARY_MODE", "lead"))  # lead | llm

    # Retrieval.
    context_max_chars: int = 9000   # ~4-5k tokens: numbers tokenise ~1 token/char, window is 8k
    top_docs: int = 5
    top_chunks: int = 8
    doc_prefilter_min_docs: int = 8

    # Local (CPU, ONNX) models used when embedder="local" / rerank="local".
    local_embed_model: str = field(default_factory=lambda: _env("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-small"))
    local_rerank_model: str = field(default_factory=lambda: _env("LOCAL_RERANK_MODEL",
                                                                 "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    local_threads: int = field(default_factory=lambda: int(_env("LOCAL_THREADS", "0")))   # 0 = auto (half the cores)
    local_max_chars: int = 2400
    rerank: str = field(default_factory=lambda: _env("RERANK", "local"))                    # off | local (cross-encoder on the top candidates)
    rerank_candidates: int = 15
    rerank_max_chars: int = 1200   # cross-encoder cost is ~linear in text length

    # PII: "off" | "mask" (rizzo-pii replaces identifiers with placeholders before embedding/storage).
    pii_mode: str = field(default_factory=lambda: _env("PII_MODE", "off"))
    pii_model: str = field(default_factory=lambda: _env("PII_MODEL", "rizzoaiacademy/rizzo-pii-0.3B"))
    pii_min_score: float = 0.5

    max_upload_mb: int = 100
    max_xlsx_cells: int = 500_000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "structrag.sqlite"

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

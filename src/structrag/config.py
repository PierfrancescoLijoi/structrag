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

    # Grounding: answers are checked against their sources or replaced by an honest refusal.
    grounding: str = field(default_factory=lambda: _env("GROUNDING", "strict"))   # strict | off
    # Calibrated on 232 questions (eval/bench/calibrate.py): every answer below relevance 0 was wrong or a non-answer,
    # the least relevant correct one scored 0.38; a lexical support gate above 0.2 only loses correct answers.
    min_relevance: float = 0.0      # reranker logit of the best passage below which we do not even ask the model
    min_support: float = 0.2        # share of a sentence's content words that its cited passage must contain
    # The 4B verifier separates right from wrong poorly (AUC 0.70-0.75 in every prompt variant tried) and rejects
    # 17-35% of correct answers to catch 4-7 of 10 wrong ones: off by default. > 0 = P("fully supported") required.
    verifier_min: float = 0.0

    # Reference graph (links.py): passages cited by the best hits join the context, one hop, never chained.
    graph: bool = field(default_factory=lambda: _env("GRAPH", "on") == "on")
    link_seeds: int = 3             # only the best hits pull their references in
    link_max: int = 3               # linked passages added per question
    link_min_relevance: float = 0.0   # reranker logit a linked passage must reach for THIS question

    # Retrieval.
    context_max_chars: int = 9000   # ~4-5k tokens: numbers tokenise ~1 token/char, window is 8k
    top_docs: int = 5
    top_chunks: int = 8
    doc_prefilter_min_docs: int = 8

    # Local (CPU, ONNX) models used when embedder="local" / rerank="local".
    local_embed_model: str = field(default_factory=lambda: _env("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-small"))
    local_rerank_model: str = field(default_factory=lambda: _env("LOCAL_RERANK_MODEL",
                                                                 "mmarco-mMiniLMv2-int8"))
    local_threads: int = field(default_factory=lambda: int(_env("LOCAL_THREADS", "0")))   # 0 = auto (half the cores)
    local_max_chars: int = 2400
    rerank: str = field(default_factory=lambda: _env("RERANK", "local"))                    # off | local (cross-encoder on the top candidates)
    rerank_candidates: int = 30
    rerank_max_chars: int = 1200   # cross-encoder cost is ~linear in text length

    # Images: OCR (RapidOCR, CPU) for figures, scanned pages and image files; optional local vision model
    # (any OpenAI-compatible server with a vision model, e.g. llama-server + SmolVLM2 / Qwen2.5-VL).
    ocr: str = field(default_factory=lambda: _env("OCR", "auto"))                    # auto | off
    ocr_lang: str = field(default_factory=lambda: _env("OCR_LANG", "latin"))         # latin | en | ch | ...
    ocr_dpi: int = 200
    ocr_min_page_chars: int = 40      # PDF pages with less native text than this are OCR'd whole
    ocr_min_figure_share: float = 0.06   # embedded PDF images below this share of the page area are ignored
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", ""))
    vision_base_url: str = field(default_factory=lambda: _env("VISION_BASE_URL", ""))

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

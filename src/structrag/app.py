"""Service container: builds the object graph once for the CLI and the web server."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chat import ChatService
from .config import Settings
from .ingest import IngestResult, Ingestor
from .llm.client import LLM, ChatModel, Embedder, get_embedder
from .parsers import SUPPORTED
from .retrieve import Retriever
from .store import Store
from .structure.profiles import OverrideStore, ProfileStore


@dataclass
class Services:
    settings: Settings
    store: Store
    llm: ChatModel
    embedder: Embedder
    ingestor: Ingestor
    retriever: Retriever
    chat: ChatService


def build_services(settings: Settings | None = None, llm: ChatModel | None = None,
                   embedder: Embedder | None = None) -> Services:
    settings = settings or Settings()
    store = Store(settings.db_path)
    llm = llm or LLM(settings)
    embedder = embedder or get_embedder(settings, llm if isinstance(llm, LLM) else None)
    ingestor = Ingestor(settings, store, embedder, llm)
    retriever = Retriever(settings, store, embedder)
    return Services(settings, store, llm, embedder, ingestor, retriever,
                    ChatService(settings, store, retriever, llm))


def scan_inbox(services: Services) -> list[tuple[str, IngestResult]]:
    """Ingest every supported file in the inbox; already-known files are skipped by hash."""
    results = []
    inbox = services.settings.inbox_dir
    inbox.mkdir(parents=True, exist_ok=True)
    for path in sorted(inbox.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            results.append((path.name, services.ingestor.ingest_file(path)))
    return results


def apply_review(services: Services, review_id: int, choice: str) -> IngestResult | None:
    """Apply a human decision: persist an override, teach the profile, rebuild the document."""
    review = services.store.get_review(review_id)
    if review is None or review["resolved"]:
        return None
    doc = services.store.conn.execute("SELECT sha256 FROM documents WHERE id=?", (review["doc_id"],)).fetchone()
    if doc is None:
        return None
    overrides = OverrideStore(services.settings.profiles_dir)
    if review["kind"] == "heading":
        level = review["level"] if choice == "A" else 0
        overrides.set(doc["sha256"], "headings", review["text"], level)
        ProfileStore(services.settings.profiles_dir).learn(
            review["format"], review["body_key"], [(review["style_key"], level)], human=True)
    else:
        overrides.set(doc["sha256"], "headers", review["text"], "ABCDEFGH".index(choice))
    services.store.resolve_review(review_id)
    return services.ingestor.reingest(review["doc_id"])


def inbox_files(settings: Settings) -> list[Path]:
    return [p for p in settings.inbox_dir.glob("*") if p.suffix.lower() in SUPPORTED]

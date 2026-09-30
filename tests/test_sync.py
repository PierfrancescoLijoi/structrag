import os
import time
from dataclasses import replace

import pytest

from structrag.app import build_services


class CountingEmbedder:
    """Wraps an embedder and records every text that actually gets embedded."""

    def __init__(self, inner):
        self.inner, self.texts = inner, []

    def embed(self, texts):
        self.texts += list(texts)
        return self.inner.embed(texts)


def _write(path, sections: dict[str, str]):
    path.write_text("".join(f"# {t}\n\n{b}\n\n" for t, b in sections.items()), encoding="utf-8")
    old = time.time() - 30                      # look "settled" to the syncer
    os.utime(path, (old, old))


@pytest.fixture()
def sv(settings):
    s = build_services(settings)
    s.ingestor.embedder = CountingEmbedder(s.ingestor.embedder)
    s.syncer.settle = 0
    s.settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    return s


def _chunks(sv):
    return [r[0] for r in sv.store.conn.execute("SELECT text FROM chunks")]


def test_edit_updates_chunks_and_reembeds_only_what_changed(sv):
    f = sv.settings.inbox_dir / "spec.md"
    body = {f"Unit {i}": f"The pressure of unit {i} is {i * 7} bar and it stays constant." for i in range(6)}
    _write(f, body)
    assert sv.syncer.run().added == ["spec.md"]
    embedded_first = len(sv.ingestor.embedder.texts)

    _write(f, {**body, "Unit 3": "The pressure of unit 3 is 999 bar after the upgrade."})
    report = sv.syncer.run()
    assert report.updated == ["spec.md"] and not report.added and not report.removed
    assert any("999 bar" in t for t in _chunks(sv)) and not any("21 bar" in t for t in _chunks(sv))
    assert len(sv.store.list_documents()) == 1                       # old version replaced, not duplicated
    assert len(sv.ingestor.embedder.texts) - embedded_first <= 4     # changed chunk + its cards, not the whole file
    assert not sv.syncer.run().changed                               # and it is stable afterwards


def test_deleted_file_removes_document_and_chunks(sv):
    f = sv.settings.inbox_dir / "gone.md"
    _write(f, {"Only": "This section will disappear soon."})
    sv.syncer.run()
    assert sv.store.conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] > 0
    f.unlink()
    assert sv.syncer.run().removed == ["gone.md"]
    assert sv.store.list_documents() == []
    assert sv.store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert sv.store.conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 0
    assert sv.retriever.search("disappear") == []


def test_rename_keeps_document_without_reprocessing(sv):
    a = sv.settings.inbox_dir / "old_name.md"
    _write(a, {"Topic": "Content that must survive a rename."})
    sv.syncer.run()
    before = len(sv.ingestor.embedder.texts)
    doc_id = sv.store.list_documents()[0]["id"]
    b = a.rename(sv.settings.inbox_dir / "new_name.md")
    assert sv.syncer.run().renamed == ["new_name.md"]
    assert sv.store.list_documents()[0]["id"] == doc_id and len(sv.ingestor.embedder.texts) == before


def test_broken_edit_keeps_old_version_and_is_not_retried(sv):
    f = sv.settings.inbox_dir / "doc.md"
    _write(f, {"Fine": "Original readable content lives here."})
    sv.syncer.run()
    _write(f, {})                                                    # emptied: nothing extractable
    first = sv.syncer.run()
    assert "doc.md" in first.failed and not first.updated
    assert any("Original readable" in t for t in _chunks(sv))        # old version still searchable
    assert not sv.syncer.run().changed                               # no retry loop until the file changes again


def test_index_settings_change_marks_documents_stale_and_rebuilds(sv, settings):
    f = sv.settings.inbox_dir / "a.md"
    _write(f, {"Topic": "Some content about pumps and valves in the plant."})
    sv.syncer.run()
    fresh = build_services(replace(settings, chunk_target_words=90))     # same DB, different chunking
    fresh.syncer.settle = 0
    assert fresh.syncer.run().rebuilt == ["a.md"]
    assert not fresh.syncer.run().changed


def test_files_still_being_written_are_left_for_next_round(sv):
    f = sv.settings.inbox_dir / "big.md"
    f.write_text("# T\n\nbody text that is still being copied\n", encoding="utf-8")   # mtime = now
    sv.syncer.settle = 60
    assert not sv.syncer.run().changed
    sv.syncer.settle = 0
    assert sv.syncer.run().added == ["big.md"]


def test_concurrent_ingest_of_same_file_is_safe(sv):
    import threading
    f = sv.settings.inbox_dir / "race.md"
    _write(f, {"Topic": "Two workers race to index this very same file."})
    results = []
    threads = [threading.Thread(target=lambda: results.append(sv.ingestor.ingest_file(f).status)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["duplicate"] * 3 + ["ingested"] and len(sv.store.list_documents()) == 1


def test_api_sync_endpoint_reports_changes(sv):
    from fastapi.testclient import TestClient
    from structrag.api import create_app
    client = TestClient(create_app(sv))
    f = sv.settings.inbox_dir / "api.md"
    _write(f, {"Topic": "Body for the API sync endpoint."})
    r = client.post("/api/inbox/scan", headers={"X-Requested-With": "t"}).json()
    assert r["added"] == ["api.md"] and r["failed"] == {}
    f.unlink()
    assert client.post("/api/inbox/scan", headers={"X-Requested-With": "t"}).json()["removed"] == ["api.md"]

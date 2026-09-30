"""Keep the index in step with the files on disk: added, edited, renamed, deleted, or stale.

Change detection is cheap: (mtime, size) is compared first and the file is only hashed when either
moved. Edited files are rebuilt through `Ingestor.reingest`, which reuses vectors of unchanged chunks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import file_sha256
from .parsers import SUPPORTED

log = logging.getLogger(__name__)
SETTLE_SECONDS = 2.0   # a file touched this recently may still be mid-write (copy, save): look again next round


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)   # index settings changed (chunking, embedder, PII mode)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.renamed or self.rebuilt or self.failed)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("added", "updated", "removed", "renamed", "rebuilt", "failed")}


class Syncer:
    def __init__(self, services, settle: float = SETTLE_SECONDS):
        self.sv, self.settle = services, settle
        self._failed: dict[str, tuple[int, int]] = {}   # path -> (mtime_ns, size) of a version that failed

    def _files(self) -> dict[str, Path]:
        inbox = self.sv.settings.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        return {str(p): p for p in sorted(inbox.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED}

    def _settled(self, st) -> bool:
        return self.settle <= 0 or time.time() - st.st_mtime >= self.settle

    def run(self) -> SyncReport:
        report, store, ing = SyncReport(), self.sv.store, self.sv.ingestor
        rows = {r["path"]: r for r in store.sync_rows()}
        files = self._files()
        missing = {p: r for p, r in rows.items() if not Path(p).exists()}

        for path_str, row in rows.items():          # known documents still on disk: edited / stale
            if path_str in missing:
                continue
            path, st = Path(path_str), Path(path_str).stat()
            if (st.st_mtime_ns, st.st_size) != (row["mtime_ns"], row["size"]) and self._settled(st):
                self._reingest(row["id"], path, report, "updated")
            elif row["index_sig"] != ing.index_sig and self._failed.get(path_str) != (st.st_mtime_ns, st.st_size):
                self._reingest(row["id"], path, report, "rebuilt")

        moved = {r["sha256"]: r for r in missing.values()}
        for path_str, path in files.items():        # new files, or a known file that was renamed / moved
            if path_str in rows:
                continue
            st = path.stat()
            if not self._settled(st) or self._failed.get(path_str) == (st.st_mtime_ns, st.st_size):
                continue
            if (old := moved.pop(file_sha256(path), None)):
                store.set_location(old["id"], path=path_str, mtime_ns=st.st_mtime_ns, size=st.st_size)
                missing.pop(old["path"])
                report.renamed.append(path.name)
                continue
            r = ing.ingest_file(path)
            if r.status in ("ingested", "needs_ocr"):
                report.added.append(path.name)
            elif r.status == "failed":
                self._failed[path_str] = (st.st_mtime_ns, st.st_size)
                report.failed[path.name] = r.error

        for path_str, row in missing.items():       # gone for good
            store.delete_document(row["id"])
            report.removed.append(Path(path_str).name)
        return report

    def _reingest(self, doc_id: int, path: Path, report: SyncReport, bucket: str) -> None:
        st = path.stat()
        r = self.sv.ingestor.reingest(doc_id)
        if r.status == "failed":          # keep the previous version searchable; do not retry until the file changes
            self._failed[str(path)] = (st.st_mtime_ns, st.st_size)
            report.failed[path.name] = r.error
            self.sv.store.set_location(doc_id, mtime_ns=st.st_mtime_ns, size=st.st_size)
            return
        self._failed.pop(str(path), None)
        getattr(report, bucket).append(path.name)

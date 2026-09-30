"""FastAPI app: upload/ingest, document browser, chat (SSE), review queue. Local-only by design."""
from __future__ import annotations

import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .app import Services, apply_review
from .parsers import SUPPORTED

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}   # blocks DNS-rebinding
CSRF_HEADER = "x-requested-with"                                     # forces a CORS preflight
UPLOAD_CHUNK = 1 << 20
MAX_JOBS_KEPT = 50
INDEX_HTML = Path(__file__).parent / "web" / "index.html"


class ChatRequest(BaseModel):
    session_id: int
    message: str = Field(min_length=1, max_length=8000)
    doc_ids: list[int] | None = None


class ReviewRequest(BaseModel):
    choice: str = Field(pattern=r"^[A-H]$")


class SessionRequest(BaseModel):
    title: str = Field(default="New chat", max_length=200)


def _hostname(raw: str) -> str:
    if raw.startswith("["):                       # IPv6 literal: [::1]:8000
        return raw.split("]")[0] + "]"
    return raw.rsplit(":", 1)[0] if ":" in raw else raw


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _safe_name(filename: str) -> str:
    p = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"[^\w.-]+", "_", Path(p).stem)[:60].strip("._") or "file"
    return f"{uuid.uuid4().hex[:8]}-{stem}{Path(p).suffix.lower()}"


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="structrag", docs_url=None, redoc_url=None)
    pool = ThreadPoolExecutor(max_workers=1)   # one ingest at a time: small GPUs, SQLite writes
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = _hostname(request.headers.get("host", ""))
        if host not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        if request.method not in ("GET", "HEAD") and CSRF_HEADER not in request.headers:
            return JSONResponse({"detail": "missing X-Requested-With header"}, status_code=403)
        return await call_next(request)

    def run_ingest(job_id: str, path: Path) -> None:
        try:
            r = services.ingestor.ingest_file(path)
            state = {"state": "done", "status": r.status, "doc_id": r.doc_id, "error": r.error,
                     "queued_for_review": r.queued_for_review, "warnings": list(r.warnings)}
        except Exception as exc:   # never let a worker die silently
            state = {"state": "error", "error": str(exc)}
        with jobs_lock:
            jobs[job_id].update(state)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/api/images/{ref}")
    def image(ref: str) -> FileResponse:
        """Thumbnail of a figure that an answer cites. `ref` is a hash: anything else is rejected (no paths)."""
        path = services.settings.data_dir / "images" / f"{ref}.jpg"
        if not re.fullmatch(r"[0-9a-f]{16}", ref) or not path.is_file():
            raise HTTPException(404, "image not found")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/health")
    def health() -> dict:
        info = services.llm.health() if hasattr(services.llm, "health") else {"ok": True}
        return {**info, "embedder": services.settings.embedder, "chat_model": services.settings.chat_model,
                "embed_model": services.settings.embed_model, "pii_mode": services.settings.pii_mode,
                "rerank": services.settings.rerank,
                "ocr": services.settings.ocr, "vision_model": services.settings.vision_model,
                "grounding": services.settings.grounding, "local_embed_model": services.settings.local_embed_model,
                "documents": len(services.store.list_documents())}

    @app.get("/api/stats")
    def stats() -> dict:
        docs = services.store.list_documents()
        pii: dict[str, int] = {}
        for d in docs:
            for label, n in d["pii"].items():
                pii[label] = pii.get(label, 0) + n
        by_status: dict[str, int] = {}
        for d in docs:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
        return {"documents": len(docs), "sections": sum(d["n_sections"] for d in docs),
                "chunks": sum(d["n_chunks"] for d in docs), "words": sum(d["words"] for d in docs),
                "by_status": by_status, "by_format": _count(d["format"] for d in docs),
                "pending_reviews": len(services.store.pending_reviews()),
                "pii": {"mode": services.settings.pii_mode, "model": services.settings.pii_model,
                        "docs_with_pii": sum(1 for d in docs if d["pii"]), "by_label": pii}}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        s = services.settings
        s.inbox_dir.mkdir(parents=True, exist_ok=True)
        queued = []
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in SUPPORTED:
                raise HTTPException(415, f"unsupported file type '{ext}'")
            dest = s.inbox_dir / _safe_name(f.filename or "file")
            size = 0
            with open(dest, "wb") as out:
                while chunk := await f.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > s.max_upload_mb * UPLOAD_CHUNK:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"file exceeds {s.max_upload_mb} MB")
                    out.write(chunk)
            job_id = uuid.uuid4().hex
            with jobs_lock:
                jobs[job_id] = {"id": job_id, "name": f.filename, "state": "running"}
                for old in list(jobs)[:-MAX_JOBS_KEPT]:
                    jobs.pop(old)
            pool.submit(run_ingest, job_id, dest)
            queued.append(job_id)
        return {"jobs": queued}

    @app.post("/api/inbox/scan")
    def inbox_scan() -> dict:
        return services.syncer.run().as_dict()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        with jobs_lock:
            return list(jobs.values())

    @app.get("/api/docs")
    def docs() -> list[dict]:
        return services.store.list_documents()

    @app.get("/api/docs/{doc_id}/outline")
    def outline(doc_id: int) -> list[dict]:
        return services.store.outline(doc_id)

    @app.post("/api/docs/{doc_id}/reingest")
    def reingest(doc_id: int) -> dict:
        r = services.ingestor.reingest(doc_id)
        return {"status": r.status, "doc_id": r.doc_id, "error": r.error}

    @app.delete("/api/docs/{doc_id}")
    def delete_doc(doc_id: int) -> dict:
        row = services.store.conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "document not found")
        services.store.delete_document(doc_id)
        path = Path(row["path"])
        inbox = services.settings.inbox_dir.resolve()
        if path.exists() and path.resolve().parent == inbox:   # never delete files outside the inbox
            path.unlink()
        return {"deleted": doc_id}

    @app.get("/api/sessions")
    def sessions() -> list[dict]:
        return services.store.list_sessions()

    @app.post("/api/sessions")
    def new_session(req: SessionRequest) -> dict:
        return {"id": services.store.create_session(req.title)}

    @app.get("/api/sessions/{session_id}")
    def session(session_id: int) -> dict:
        s = services.store.get_session(session_id)
        if s is None:
            raise HTTPException(404, "session not found")
        return {**s, "messages": services.store.messages(session_id)}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: int) -> dict:
        services.store.delete_session(session_id)
        return {"deleted": session_id}

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        if services.store.get_session(req.session_id) is None:
            raise HTTPException(404, "session not found")

        def events():
            for ev in services.chat.ask(req.session_id, req.message, req.doc_ids):
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/review")
    def review_queue() -> list[dict]:
        return services.store.pending_reviews()

    @app.post("/api/review/{review_id}")
    def review(review_id: int, req: ReviewRequest) -> dict:
        r = apply_review(services, review_id, req.choice)
        if r is None:
            raise HTTPException(404, "review item not found or already resolved")
        return {"status": r.status, "doc_id": r.doc_id}

    return app

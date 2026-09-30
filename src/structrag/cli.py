"""Command line entry point: ingest, scan, watch, ask, serve, docs, review, doctor."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from .app import Services, build_services, scan_inbox
from .config import Settings
from .parsers import SUPPORTED

WATCH_INTERVAL = 5.0


def _iter_files(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        yield from (sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p])


def cmd_ingest(sv: Services, args) -> int:
    failed = 0
    for path in _iter_files(args.paths):
        r = sv.ingestor.ingest_file(path)
        detail = r.error or f"{r.n_sections} sections, {r.n_chunks} chunks, {r.strategy} ({r.confidence:.2f})"
        print(f"{r.status:10} {path.name}  {detail}" + (f"  [{r.queued_for_review} for review]" if r.queued_for_review else ""))
        failed += r.status == "failed"
    return 1 if failed else 0


def cmd_scan(sv: Services, args) -> int:
    for name, r in scan_inbox(sv):
        print(f"{r.status:10} {name}")
    return 0


def _watch_loop(sv: Services, interval: float, stop: threading.Event) -> None:
    while not stop.is_set():
        for name, r in scan_inbox(sv):
            if r.status not in ("duplicate",):
                logging.getLogger("structrag.watch").info("%s: %s", name, r.status)
        stop.wait(interval)


def cmd_watch(sv: Services, args) -> int:
    stop = threading.Event()
    try:
        _watch_loop(sv, args.interval, stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


def cmd_ask(sv: Services, args) -> int:
    sid = sv.store.create_session()
    for ev in sv.chat.ask(sid, args.question):
        if ev["type"] == "sources":
            for s in ev["sources"]:
                print(f"[{s['n']}] {s['doc']} > {s['section']} ({s['loc']})", file=sys.stderr)
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "error":
            print(f"\nerror: {ev['text']}", file=sys.stderr)
    print()
    return 0


def cmd_serve(sv: Services, args) -> int:
    import uvicorn

    from .api import create_app
    stop = threading.Event()
    threading.Thread(target=_watch_loop, args=(sv, WATCH_INTERVAL, stop), daemon=True).start()
    print(f"structrag dashboard on http://127.0.0.1:{args.port}")
    try:
        uvicorn.run(create_app(sv), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        stop.set()
    return 0


def cmd_docs(sv: Services, args) -> int:
    for d in sv.store.list_documents():
        print(f"{d['id']:4} {d['status']:9} {d['format']:5} {d['n_sections']:4} sec {d['n_chunks']:5} chunks  "
              f"{d['strategy'] or '-':22} {d['confidence']:.2f}  pii={sum(d['pii'].values()):3}  {d['title']}")
    return 0


def cmd_review(sv: Services, args) -> int:
    for r in sv.store.pending_reviews():
        probs = ", ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
        print(f"#{r['id']} [{r['kind']}] {r['doc_title']}: {r['text'][:80]!r}  ({probs})")
    return 0


def cmd_doctor(sv: Services, args) -> int:
    info = sv.llm.health() if hasattr(sv.llm, "health") else {"ok": True}
    print(info, f"pii_mode={sv.settings.pii_mode}")
    if not info.get("ok"):
        print("model server unreachable: start Ollama or set STRUCTRAG_BASE_URL / STRUCTRAG_EMBEDDER=hash")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="structrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    sub.add_parser("scan").set_defaults(fn=cmd_scan)
    p = sub.add_parser("watch"); p.add_argument("--interval", type=float, default=WATCH_INTERVAL); p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("ask"); p.add_argument("question"); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000); p.set_defaults(fn=cmd_serve)
    sub.add_parser("docs").set_defaults(fn=cmd_docs)
    sub.add_parser("review").set_defaults(fn=cmd_review)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(build_services(Settings()), args)


if __name__ == "__main__":
    raise SystemExit(main())

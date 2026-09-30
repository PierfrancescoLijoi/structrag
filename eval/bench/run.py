"""Benchmark on real public documents: retrieval hit@k and end-to-end answer accuracy.

    PYTHONPATH=src python eval/bench/run.py --tag hash --embedder hash
    PYTHONPATH=src python eval/bench/run.py --tag local --embedder local --answer

Questions live in qa.json. A retrieval *hit* means one retrieved chunk contains ALL evidence strings
(so a table fact needs its row/column context in the same chunk). An answer is *correct* when the
generated text contains every string of at least one accepted alternative; unanswerable questions
are correct only if the model abstains.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE / "docs"
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
ABSTAIN = re.compile(
    r"(not (contain|mention|provide|include|specif|state|find|available|addressed)|no (information|mention|data)|"
    r"does(n't| not) (contain|mention|say|specify|provide|state)|cannot (be )?(answer|find|determine)|can't (answer|find)|"
    r"unable to|isn't (mentioned|stated|specified)|is not (mentioned|stated|specified)|insufficient|"
    r"non (contiene|è (indicat|present|riportat|specificat))|non ci sono|non ho trovato|non risulta|nessuna informazione|"
    r"non è possibile)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)            # 1,720,320 -> 1720320
    s = s.replace("−", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def has_all(text: str, needles: list[str]) -> bool:
    t = norm(text)
    return all(norm(n) in t for n in needles)


def answer_ok(qa: dict, text: str) -> bool:
    if qa["answer"] is None:
        return bool(ABSTAIN.search(text))
    return any(has_all(text, alt) for alt in qa["answer"])


def build(args):
    sys.path.insert(0, str(HERE.parent.parent / "src"))
    from structrag.app import build_services
    from structrag.config import Settings
    from structrag.llm.client import LLM

    base = Settings()
    extra = {"local_embed_model": args.embed_model} if args.embed_model else {}
    if args.rerank_model:
        extra["local_rerank_model"] = args.rerank_model
    s = Settings(data_dir=CACHE / (args.cache or args.tag) / "data", profiles_dir=CACHE / (args.cache or args.tag) / "profiles",
                 embedder=args.embedder, base_url=args.llm or base.base_url, rerank=args.rerank, rerank_candidates=args.cand, rerank_max_chars=args.rr_chars, **extra)
    llm = LLM(s) if (args.answer or args.agent) else None
    services = build_services(s, llm=llm or _NoLLM())
    if not args.agent:
        services.ingestor.llm = None
    return services


class _NoLLM:
    def chat(self, *a, **k): raise RuntimeError("LLM disabled")
    stream = choose = chat


def ingest_all(services) -> dict:
    stats = {}
    have = {Path(d["path"]).name for d in services.store.list_documents()}
    for f in sorted(DOCS.iterdir()):
        if f.name in have:
            continue
        t = time.time()
        r = services.ingestor.ingest_file(f)
        stats[f.name] = {"status": r.status, "chunks": r.n_chunks, "sec": round(time.time() - t, 1)}
        print(f"  ingest {f.name:45} {r.status:9} {r.n_chunks:5} chunks {stats[f.name]['sec']:6.1f}s", flush=True)
    return stats


def evidence_exists(services, qa: dict) -> bool:
    rows = services.store.conn.execute(
        "SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.path LIKE ?", (f"%{qa['doc']}",))
    return any(has_all(r[0], qa["evidence"]) for r in rows)


def run(args) -> dict:
    services = build(args)
    ingest_all(services)
    qas = json.loads((HERE / "qa.json").read_text(encoding="utf-8"))
    if args.only:
        qas = [q for q in qas if re.search(args.only, q["id"])]
    ks = [int(k) for k in args.k.split(",")]
    kmax = max(ks)
    rows, missing = [], []
    for qa in qas:
        row = {"id": qa["id"], "type": qa["type"], "doc": qa["doc"], "q": qa["q"]}
        if qa["evidence"]:
            if not evidence_exists(services, qa):
                missing.append(qa["id"])
            t = time.time()
            hits = services.retriever.search(qa["q"], None, kmax)
            row["latency"] = round(time.time() - t, 3)
            row["rank"] = next((i + 1 for i, h in enumerate(hits) if has_all(h.text, qa["evidence"])), None)
            row["doc_rank"] = next((i + 1 for i, h in enumerate(hits) if Path(h.doc_title).name and
                                    qa["doc"].split(".")[0].lower() in Path(_path(services, h.doc_id)).stem.lower()), None)
            ctx = services.retriever.build_context(hits[:services.settings.top_chunks])
            row["ctx_hit"] = any(has_all(c.text, qa["evidence"]) for c in ctx)
        if args.answer:
            row["answer"] = _ask(services, qa["q"])
            row["correct"] = answer_ok(qa, row["answer"])
        rows.append(row)
    report(rows, ks, missing, args)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"rows": rows}


_paths: dict[int, str] = {}


def _path(services, doc_id: int) -> str:
    if doc_id not in _paths:
        r = services.store.conn.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
        _paths[doc_id] = r[0]
    return _paths[doc_id]


def _ask(services, question: str) -> str:
    sid = services.store.create_session()
    text = []
    for ev in services.chat.ask(sid, question):
        if ev["type"] == "token":
            text.append(ev["text"])
        elif ev["type"] == "error":
            text.append(f"[error] {ev['text']}")
    return "".join(text).strip()


def report(rows, ks, missing, args) -> None:
    ev = [r for r in rows if "rank" in r]
    print(f"\n== {args.tag}: {len(rows)} questions, {len(ev)} with evidence ==")
    if missing:
        print(f"!! evidence not found in any chunk (parse/chunk/QA issue): {missing}")
    for k in ks:
        print(f"hit@{k:<2} chunk: {sum(1 for r in ev if r['rank'] and r['rank'] <= k) / len(ev):6.1%}   "
              f"doc: {sum(1 for r in ev if r['doc_rank'] and r['doc_rank'] <= k) / len(ev):6.1%}")
    print(f"context hit (as shown to LLM): {sum(r['ctx_hit'] for r in ev) / len(ev):6.1%}")
    lat = sorted(r["latency"] for r in ev)
    print(f"retrieval latency: median {lat[len(lat) // 2] * 1000:.0f} ms, p95 {lat[int(len(lat) * .95)] * 1000:.0f} ms")
    by = defaultdict(list)
    for r in ev:
        by[r["type"]].append(r)
    for t, rs in sorted(by.items()):
        k = ks[len(ks) // 2]
        print(f"  {t:12} n={len(rs):3} hit@{k} {sum(1 for r in rs if r['rank'] and r['rank'] <= k) / len(rs):6.1%}")
    if args.answer:
        ok = sum(r["correct"] for r in rows)
        print(f"ANSWER ACCURACY: {ok}/{len(rows)} = {ok / len(rows):.1%}")
        for t in sorted({r["type"] for r in rows}):
            rs = [r for r in rows if r["type"] == t]
            print(f"  {t:12} n={len(rs):3} acc {sum(r['correct'] for r in rs) / len(rs):6.1%}")
    k = ks[len(ks) // 2]
    miss = [r["id"] for r in ev if not (r["rank"] and r["rank"] <= k)]
    print(f"retrieval misses @{k}: {miss}")
    if args.answer:
        print("wrong answers:", [r["id"] for r in rows if not r["correct"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hash")
    p.add_argument("--embedder", default="hash")
    p.add_argument("--cache", default=None, help="reuse the index built under another tag")
    p.add_argument("--rr-chars", type=int, default=1200)
    p.add_argument("--cand", type=int, default=15, help="rerank candidates")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--rerank", default="off", choices=["off", "local"])
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--llm", default=None, help="OpenAI-compatible base url")
    p.add_argument("--answer", action="store_true")
    p.add_argument("--agent", action="store_true", help="use the LLM for structure inference during ingest")
    p.add_argument("--k", default="1,3,5,8")
    p.add_argument("--only", default=None, help="regex on question id")
    run(p.parse_args())

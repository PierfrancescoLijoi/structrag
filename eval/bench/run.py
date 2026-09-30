"""Benchmark on real public documents: retrieval hit@k, answer accuracy, refusals and hallucinations.

    PYTHONPATH=src python eval/bench/run.py --tag t --cache all --answer --llm http://127.0.0.1:8081/v1 \
        --docs docs,docs_blind,docs_blind2 --qa qa.json,qa_blind.json,qa_blind2.json,qa_traps.json

All documents go into ONE index (a realistic multi-document corpus). Every question is asked against all of it.

Per question:
  answerable   -> correct | wrong (answered, incorrect) | refused
  unanswerable -> refused (right) | hallucinated (answered anyway)
A retrieval *hit* = one retrieved chunk contains ALL evidence strings. An answer is *correct* when it contains
every string of one accepted alternative. `cite_ok` = the cited passages contain the evidence, or the accepted answer
(the same fact can be stated in two chunks; footnote markers like `[2]` inside a number are ignored).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
ABSTAIN = re.compile(
    r"(not (contain|mention|provide|include|specif|state|find|available|addressed)|no (information|mention|data)|"
    r"does(n't| not) (contain|mention|say|specify|provide|state)|cannot (be )?(answer|find|determine)|can't (answer|find)|"
    r"unable to|isn't (mentioned|stated|specified)|is not (mentioned|stated|specified)|insufficient|"
    r"could not find enough|non (contiene|è (indicat|present|riportat|specificat))|non ci sono|non ho trovato|"
    r"non risulta|nessuna informazione|non è possibile)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)            # 1,720,320 -> 1720320
    s = s.replace("−", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def has_all(text: str, needles: list[str]) -> bool:
    t = norm(text)
    return all(norm(n) in t for n in needles)


def strip_cites(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text)


def answer_ok(qa: dict, text: str) -> bool:
    return any(has_all(strip_cites(text), alt) for alt in qa["answer"])


def parse_overrides(pairs: list[str], settings_cls) -> dict:
    types = {f.name: f.type for f in dataclasses.fields(settings_cls)}
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        kind = str(types[key])
        out[key] = (float(value) if "float" in kind else int(value) if "int" in kind
                    else value in ("on", "true", "1") if "bool" in kind else value)
    return out


def build(args):
    sys.path.insert(0, str(HERE.parent.parent / "src"))
    import structrag.chat as chat_mod
    from structrag.app import build_services
    from structrag.config import Settings
    from structrag.llm.client import LLM

    base = Settings()
    cache = CACHE / (args.cache or args.tag)
    extra = {"local_embed_model": args.embed_model} if args.embed_model else {}
    if args.rerank_model:
        extra["local_rerank_model"] = args.rerank_model
    if args.vision:
        extra.update(vision_model="vision", vision_base_url=args.vision)
    extra.update(parse_overrides(args.set, Settings))
    s = Settings(data_dir=cache / "data", profiles_dir=cache / "profiles", embedder=args.embedder,
                 base_url=args.llm or base.base_url, rerank=args.rerank, rerank_candidates=args.cand,
                 rerank_max_chars=args.rr_chars, **extra)
    services = build_services(s, llm=LLM(s) if (args.answer or args.agent) else _NoLLM())
    if not args.agent:
        services.ingestor.llm = None
    # instrumentation: full text of cited passages, and the model's raw answer before grounding
    original_source, original_grounded = chat_mod._source, services.chat._grounded_answer
    chat_mod._source = lambda c, cited=False, evidence="": {**original_source(c, cited, evidence), "full": c.text}

    def grounded(question, contexts, raw):
        checked, verdict = original_grounded(question, contexts, raw)
        verdict["raw"] = raw
        verdict["all_claims"] = [{"text": c.text, "support": c.support, "cites": list(c.cites)} for c in checked.claims]
        return checked, verdict
    services.chat._grounded_answer = grounded
    return services


class _NoLLM:
    def chat(self, *a, **k): raise RuntimeError("LLM disabled")
    stream = choose = chat


def ingest_all(services, dirs: list[Path]) -> None:
    have = {Path(d["path"]).name for d in services.store.list_documents()}
    for docs in dirs:
        for f in sorted(docs.iterdir()):
            if f.name in have or not f.is_file():
                continue
            t = time.time()
            r = services.ingestor.ingest_file(f)
            print(f"  ingest {f.name:45} {r.status:9} {r.n_chunks:5} chunks {time.time() - t:6.1f}s", flush=True)


def evidence_exists(services, qa: dict) -> bool:
    rows = services.store.conn.execute(
        "SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.path LIKE ?", (f"%{qa['doc']}",))
    return any(has_all(r[0], qa["evidence"]) for r in rows)


def ask(services, question: str) -> dict:
    sid = services.store.create_session()
    final, t, offered = {}, time.time(), 0
    for ev in services.chat.ask(sid, question):
        if ev["type"] == "sources":
            offered = sum(bool(s["via"]) for s in ev["sources"])
        if ev["type"] == "final":
            final = ev
        elif ev["type"] == "error":
            final = {"answered": False, "text": f"[error] {ev['text']}", "citations": [], "verdict": {}}
    final["seconds"] = round(time.time() - t, 2)
    final["linked_offered"] = offered
    return final


def run(args) -> list[dict]:
    services = build(args)
    ingest_all(services, [HERE / d for d in args.docs.split(",")])
    if args.ingest_only:
        return []
    qas = []
    for name in args.qa.split(","):
        for q in json.loads((HERE / name).read_text(encoding="utf-8")):
            qas.append({**q, "set": Path(name).stem})
    if args.only:
        qas = [q for q in qas if re.search(args.only, q["id"])]
    ks = [int(k) for k in args.k.split(",")]
    rows, missing = [], []
    for qa in qas:
        row = {"id": qa["id"], "set": qa["set"], "type": qa["type"], "doc": qa["doc"], "q": qa["q"]}
        if qa["evidence"]:
            if not evidence_exists(services, qa):
                missing.append(qa["id"])
            t = time.time()
            hits = services.retriever.search(qa["q"], None, max(ks))
            row["latency"] = round(time.time() - t, 3)
            row["rank"] = next((i + 1 for i, h in enumerate(hits) if has_all(h.text, qa["evidence"])), None)
            top = hits[:services.settings.top_chunks]
            row["ctx_hit_plain"] = any(has_all(c.text, qa["evidence"]) for c in services.retriever.build_context(top))
            ctx = services.retriever.build_context(top, linked=services.retriever.follow_links(qa["q"], top))
            row["ctx_hit"] = any(has_all(c.text, qa["evidence"]) for c in ctx)
            row["ctx_via_link"] = any(c.via and has_all(c.text, qa["evidence"]) for c in ctx)
        if args.answer:
            final = ask(services, qa["q"])
            answered = bool(final.get("answered")) and not ABSTAIN.search(final.get("text", ""))
            row.update(answered=answered, answer=final.get("text", ""), seconds=final["seconds"],
                       verdict=final.get("verdict", {}), n_cites=len(final.get("citations", [])),
                       linked_offered=final["linked_offered"],
                       linked_cited=sum(bool(c.get("via")) for c in final.get("citations", [])))
            if qa["answer"] is None:
                row["outcome"] = "hallucinated" if answered else "refused"
            elif not answered:
                row["outcome"] = "refused"
            else:
                row["outcome"] = "correct" if answer_ok(qa, row["answer"]) else "wrong"
            if row["outcome"] == "correct" and qa["evidence"]:
                cited = " ".join(c.get("full", c["text"]) for c in final.get("citations", []))
                row["cite_ok"] = has_all(strip_cites(cited), qa["evidence"]) or answer_ok(qa, cited)
        rows.append(row)
        print(f"  q {len(rows)}/{len(qas)} {qa['id']:8} {row.get('outcome', ''):12} {row.get('seconds', '')}", flush=True)
    report(rows, ks, missing, args)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


def pct(n: int, d: int) -> str:
    return f"{n / d:6.1%}" if d else "   n/a"


def report(rows, ks, missing, args) -> None:
    ev = [r for r in rows if "rank" in r]
    print(f"\n== {args.tag}: {len(rows)} questions, {len(ev)} with retrieval evidence ==")
    if missing:
        print(f"!! evidence not found in any chunk (parse/chunk/QA issue): {missing}")
    if ev:
        print("retrieval  " + "  ".join(f"hit@{k} {pct(sum(1 for r in ev if r['rank'] and r['rank'] <= k), len(ev))}" for k in ks)
              + f"  | context hit {pct(sum(r['ctx_hit'] for r in ev), len(ev))}"
              + f" (without graph {pct(sum(r['ctx_hit_plain'] for r in ev), len(ev))},"
              + f" reached only through a link: {sum(r['ctx_via_link'] and not r['ctx_hit_plain'] for r in ev)})"
              + f"  | median {sorted(r['latency'] for r in ev)[len(ev) // 2] * 1000:.0f} ms")
    if not args.answer:
        return
    print(f"\n{'set':14} {'n':>4} {'correct':>8} {'wrong':>7} {'false-refusal':>14} {'unans.refused':>14} {'halluc.':>8} {'cite_ok':>8}")
    for name in [*sorted({r['set'] for r in rows}), "ALL"]:
        rs = [r for r in rows if name == "ALL" or r["set"] == name]
        ans = [r for r in rs if r["type"] != "unanswerable"]
        un = [r for r in rs if r["type"] == "unanswerable"]
        c = sum(r["outcome"] == "correct" for r in ans)
        cited = [r for r in ans if "cite_ok" in r]
        print(f"{name:14} {len(rs):4} {pct(c, len(ans)):>8} {pct(sum(r['outcome'] == 'wrong' for r in ans), len(ans)):>7} "
              f"{pct(sum(r['outcome'] == 'refused' for r in ans), len(ans)):>14} "
              f"{pct(sum(r['outcome'] == 'refused' for r in un), len(un)):>14} "
              f"{sum(r['outcome'] == 'hallucinated' for r in un):>3}/{len(un):<4} {pct(sum(r['cite_ok'] for r in cited), len(cited)):>8}")
    answered = [r for r in rows if r["answered"]]
    good = sum(r["outcome"] == "correct" for r in answered)
    print(f"\nprecision of answers given: {pct(good, len(answered))} ({good}/{len(answered)})   "
          f"mean answer time {sum(r['seconds'] for r in rows) / len(rows):.1f}s")
    offered = [r for r in rows if r["linked_offered"]]
    print(f"graph: linked passages offered in {pct(len(offered), len(rows))} of questions ({len(offered)}), "
          f"cited in {sum(bool(r['linked_cited']) for r in rows)}; outcome when offered: "
          f"{ {o: sum(r['outcome'] == o for r in offered) for o in ('correct', 'wrong', 'refused', 'hallucinated')} }")
    bad = [f"{r['id']}:{r['outcome']}" for r in rows if r["outcome"] in ("wrong", "hallucinated", "refused")
           and not (r["outcome"] == "refused" and r["type"] == "unanswerable")]
    print("problems:", bad)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hash")
    p.add_argument("--docs", default="docs", help="comma-separated folders under eval/bench")
    p.add_argument("--qa", default="qa.json", help="comma-separated question files")
    p.add_argument("--embedder", default="local")
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    p.add_argument("--rerank", default="local", choices=["off", "local"])
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--cache", default=None, help="reuse the index built under another tag")
    p.add_argument("--cand", type=int, default=30, help="rerank candidates")
    p.add_argument("--rr-chars", type=int, default=1200)
    p.add_argument("--llm", default=None, help="OpenAI-compatible base url of the chat model")
    p.add_argument("--vision", default=None, help="base url of a vision model (figure descriptions)")
    p.add_argument("--answer", action="store_true")
    p.add_argument("--ingest-only", action="store_true", help="build the index and stop")
    p.add_argument("--agent", action="store_true", help="use the LLM for structure inference during ingest")
    p.add_argument("--k", default="1,3,5,8")
    p.add_argument("--only", default=None, help="regex on question id")
    p.add_argument("--set", action="append", default=[], help="Settings override, e.g. --set min_relevance=-6")
    run(p.parse_args())

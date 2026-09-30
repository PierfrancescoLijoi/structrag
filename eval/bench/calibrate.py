"""Choose the grounding thresholds from a permissive run (results/calib.json).

The run records, per question: best-passage relevance, each sentence's lexical support, and the LLM verifier's
P(supported). Here we replay the gates offline for every (relevance, support, verifier) combination and report
what each would have done: correct answers kept, wrong answers, hallucinations, false refusals.

Selection uses one half of the questions; the other half validates the choice (no peeking).
    python eval/bench/calibrate.py results/calib.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import answer_ok  # noqa: E402

RELEVANCE = [-1000, -10, -9, -8, -7, -6, -5, -4, -3]
SUPPORT = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
VERIFIER = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
WRONG_COST = 3          # a wrong or invented answer costs as much as three correct ones gained


def load(path: str, qa_files: list[str]):
    qas = {}
    for name in qa_files:
        for q in json.loads((HERE / name).read_text(encoding="utf-8")):
            qas[q["id"]] = q
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return rows, qas


def replay(row: dict, qa: dict, r_min: float, s_min: float, v_min: float) -> str:
    """correct | wrong | halluc | refused_ok | refused_bad for one question under the given thresholds."""
    unanswerable = qa["answer"] is None
    v = row.get("verdict", {})
    relevance = v.get("relevance")
    claims = [c for c in v.get("all_claims", []) if c["support"] >= s_min]
    answered = bool(claims) and (relevance is None or relevance >= r_min) and v.get("verifier", 1.0) >= v_min
    if not answered:
        return "refused_ok" if unanswerable else "refused_bad"
    if unanswerable:
        return "halluc"
    return "correct" if answer_ok(qa, " ".join(c["text"] for c in claims)) else "wrong"


def evaluate(rows, qas, params, subset) -> dict:
    counts = {"correct": 0, "wrong": 0, "halluc": 0, "refused_ok": 0, "refused_bad": 0}
    for row in rows:
        if subset(row):
            counts[replay(row, qas[row["id"]], *params)] += 1
    counts["score"] = counts["correct"] - WRONG_COST * (counts["wrong"] + counts["halluc"])
    return counts


def main(path: str) -> None:
    rows, qas = load(path, ["qa.json", "qa_blind.json", "qa_blind2.json", "qa_traps.json"])
    rows = [r for r in rows if "verdict" in r and r["id"] in qas]
    fit = lambda r: hash(r["id"]) % 2 == 0 if False else sum(map(ord, r["id"])) % 2 == 0     # stable split
    hold = lambda r: not fit(r)
    grid = list(itertools.product(RELEVANCE, SUPPORT, VERIFIER))
    ranked = sorted(grid, key=lambda p: -evaluate(rows, qas, p, fit)["score"])
    print(f"{len(rows)} questions ({sum(map(fit, rows))} fit / {sum(map(hold, rows))} holdout)")
    print("baseline = no gates (R=-1000, S=0, V=0):")
    for name, sub in (("fit", fit), ("holdout", hold)):
        print("  ", name, evaluate(rows, qas, (-1000, 0.0, 0.0), sub))
    print("\nbest on fit -> holdout")
    for p in ranked[:8]:
        print(f"  R={p[0]:>6} S={p[1]:.1f} V={p[2]:.2f}  fit {evaluate(rows, qas, p, fit)}\n{'':30}holdout {evaluate(rows, qas, p, hold)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "calib.json"))

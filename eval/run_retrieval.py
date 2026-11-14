"""Recall@5 and MRR@10 per setup on one split of eval/questions.jsonl, saved to eval/results/.

Tune on dev only. Run --split test once, on the final setup. --rerankers adds the slow setups.
"""

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path

from lotse.chunking import passage
from lotse.config import RERANKER_MODEL, ROOT, SEARCH_CANDIDATES, SMALL_RERANKER_MODEL
from lotse.metrics import has_all_gold, reciprocal_rank, regressions, wilson_interval
from lotse.retrieval import bm25, dense, hybrid
from lotse.retrieval.index import Index, load_index
from lotse.retrieval.rerank import score

QUESTIONS = Path(__file__).with_name("questions.jsonl")
RESULTS = ROOT / "eval" / "results"
BASELINE = Path(__file__).with_name("baseline.json")
GATED = "hybrid_rrf"  # no reranker, so CI can run it on a CPU in about a minute
RERANKERS = {"bge": RERANKER_MODEL, "small": SMALL_RERANKER_MODEL}

Ranker = Callable[[Index, dict], list[int]]


def reranked(model: str) -> Ranker:
    def rank(index: Index, question: dict) -> list[int]:
        positions = hybrid.search(index, question["question"], question["vector"])
        passages = [passage(index.chunks[p]) for p in positions]
        scores = score(question["question"], passages, model)
        return [p for _, p in sorted(zip(scores, positions), key=lambda pair: -pair[0])]

    return rank


def evaluate(index: Index, questions: list[dict], ranker: Ranker) -> dict:
    hits, ranks, per_type = 0, [], {}
    for question in questions:
        chunks = [index.chunks[p] for p in ranker(index, question)]
        hit = has_all_gold(chunks[:5], question["gold"])
        hits += hit
        ranks.append(reciprocal_rank(chunks[:10], question["gold"]))
        if not hit:
            question.setdefault("missed_by", []).append(ranker.__name__)
        found, total = per_type.get(question["type"], (0, 0))
        per_type[question["type"]] = (found + hit, total + 1)
    low, high = wilson_interval(hits, len(questions))
    return {
        "recall_at_5": round(hits / len(questions), 3),
        "recall_interval_95": [round(low, 3), round(high, 3)],
        "mrr_at_10": round(statistics.fmean(ranks), 3),
        "n": len(questions),
        "recall_by_type": {kind: f"{found}/{n}" for kind, (found, n) in per_type.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--rerankers", nargs="*", default=[], choices=sorted(RERANKERS))
    parser.add_argument("--gate", action="store_true", help="fail if below eval/baseline.json")
    parser.add_argument("--save-baseline", action="store_true", help="store this run as it")
    args = parser.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["split"] == args.split and q["answerable"]]
    index = load_index()
    vectors = dense.embed([q["question"] for q in questions])
    for question, vector in zip(questions, vectors):
        question["vector"] = vector

    def bm25_only(index: Index, question: dict) -> list[int]:
        return bm25.search(index.bm25, question["question"], SEARCH_CANDIDATES)

    def dense_only(index: Index, question: dict) -> list[int]:
        return dense.search(index.embeddings, question["vector"], SEARCH_CANDIDATES)

    def hybrid_rrf(index: Index, question: dict) -> list[int]:
        return hybrid.search(index, question["question"], question["vector"])

    rankers: list[Ranker] = [bm25_only, dense_only, hybrid_rrf]
    for name in args.rerankers:
        rankers.append(reranked(RERANKERS[name]))
        rankers[-1].__name__ = f"hybrid_rerank_{name}"

    setups = {ranker.__name__: evaluate(index, questions, ranker) for ranker in rankers}
    for name, metrics in setups.items():
        print(f"{name:22} {json.dumps(metrics, ensure_ascii=False)}")

    git = ["git", "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, cwd=ROOT, capture_output=True, text=True, check=False)
    commit = commit.stdout.strip()
    result = {
        "date": date.today().isoformat(),
        "commit": commit or "uncommitted",
        "split": args.split,
        "chunks": len(index.chunks),
        "origins": sorted({q["origin"] for q in questions}),
        "setups": setups,
        "missed": {q["id"]: q["missed_by"] for q in questions if "missed_by" in q},
    }
    path = RESULTS / f"{result['date']}_{result['commit']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(ROOT)}")

    measured = {name: setups[GATED][name] for name in ("recall_at_5", "mrr_at_10")}
    if args.save_baseline:
        kept = ("date", "commit", "split", "chunks", "origins")
        baseline = {"setup": GATED, "metrics": measured, "n": len(questions)}
        baseline |= {name: result[name] for name in kept}
        BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.gate:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        if (baseline["chunks"], baseline["n"]) != (len(index.chunks), len(questions)):
            raise SystemExit("The index or the questions changed: measure a new baseline")
        if failed := regressions(measured, baseline):
            raise SystemExit("Quality gate failed\n" + "\n".join(failed))
        print(f"Quality gate passed: {measured} against {baseline['metrics']}")


if __name__ == "__main__":
    main()

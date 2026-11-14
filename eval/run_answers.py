"""Answer metrics on one split of eval/questions.jsonl, saved to eval/results/.

The pipeline itself gives the answer rate, the citation validity and the refusal rate. A second
model judges correctness against the reference answer, and whether each sentence is supported by
the passages it names. Tune on dev only. Run --split test once, on the final setup.

The answer model is not repeatable, so compare two setups with --runs 3, never on one run each.
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lotse import config
from lotse.chunking import passage
from lotse.generation import llm, prompts
from lotse.generation.answer import MARKER, sentences
from lotse.metrics import over_runs
from lotse.pipeline import Ranked, Result, answer, index, retrieve

HERE = Path(__file__).parent
RESULTS = config.ROOT / "eval" / "results"
DETAILS = config.ROOT / "data" / "eval_answers.jsonl"  # full texts, to check the judge by hand
CACHE = config.INDEX_DIR / "reranked.json"
# USD per million tokens, input and output: list prices read on 2026-11-13.
PRICES = {"mistral-small-2603": (0.15, 0.60), "mistral-large-4-0": (0.70, 2.10)}

JUDGE_TEXT = (HERE / "judge_prompt.md").read_text(encoding="utf-8")
_sections = re.split(r"^## (\w+)\n", JUDGE_TEXT, flags=re.MULTILINE)  # intro, name, body, name, ...
JUDGE = dict(zip(_sections[1::2], _sections[2::2]))


class Verdict(BaseModel):
    reason: str  # asked for first, so the judge argues before it decides
    verdict: Literal["correct", "partly", "wrong"]


class Support(BaseModel):
    sentence: int
    reason: str
    supported: bool


class Supports(BaseModel):
    sentences: list[Support]


def ranked_passages(questions: list[dict]) -> dict[str, Ranked]:
    """Retrieval takes about 25 s a question on a CPU and is the same in every run, so its result
    is kept next to the index until the index or a retrieval setting changes."""
    meta = (config.INDEX_DIR / "meta.json").read_text(encoding="utf-8")
    settings = [config.RERANKER_MODEL, config.RERANK_CANDIDATES, config.SEARCH_CANDIDATES]
    settings += [config.RRF_K, config.ANSWER_PASSAGES, sorted(config.API_DOCS)]
    version = hashlib.sha256((meta + json.dumps(settings)).encode()).hexdigest()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if cache.get("version") != version:
        cache = {"version": version, "ranked": {}}
    chunks = {chunk["id"]: chunk for chunk in index().chunks}
    for number, question in enumerate(questions, start=1):
        text = question["question"]
        if text not in cache["ranked"]:
            print(f"retrieving {number}/{len(questions)}", end="\r", flush=True)
            cache["ranked"][text] = [
                (score, chunk["id"]) for score, chunk in retrieve(index(), text)
            ]
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {
        text: [(score, chunks[chunk_id]) for score, chunk_id in pairs]
        for text, pairs in cache["ranked"].items()
    }


def timed_answer(question: str, ranked: Ranked) -> tuple[Result, float]:
    seconds = []

    def complete(messages: list[dict]):
        start = time.perf_counter()
        reply = llm.complete(messages)
        seconds.append(time.perf_counter() - start)
        return reply

    return answer(question, ranked, complete), sum(seconds)


def judged(kind: str, content: str, schema: type[BaseModel]):
    system = {"role": "system", "content": JUDGE[kind].strip()}
    return llm.judge([system, {"role": "user", "content": content}], schema)


def grade(question: dict, result: Result) -> dict:
    """The judge's view of one shown answer: right against the reference, and backed by the
    passages each sentence names."""
    asked = f"Frage: {question['question']}\n\nReferenzantwort: {question['reference_answer']}"
    verdict = judged("correctness", f"{asked}\n\nAntwort des Systems: {result.text}", Verdict)

    parts = sentences(result.text)
    cited = sorted({int(n) for group in MARKER.findall(result.text) for n in group.split(",")})
    sources = "\n\n".join(f"[{n}] {passage(result.passages[n - 1])}" for n in cited)
    numbered = "\n".join(f"{n}. {part}" for n, part in enumerate(parts, start=1))
    supports = judged("support", f"Quellen:\n\n{sources}\n\nSätze:\n{numbered}", Supports)
    backed = {s.sentence for s in supports.sentences if s.supported} if supports else set()
    return {
        "verdict": verdict.verdict if verdict else None,
        "verdict_reason": verdict.reason if verdict else None,
        "sentences": len(parts) if supports else 0,  # 0: the judge's reply was unusable
        "supported": len(backed & set(range(1, len(parts) + 1))),
        "support": [s.model_dump() for s in supports.sentences] if supports else None,
    }


def answered(row: dict) -> bool:
    return row["status"] == "answered"


def metrics(rows: list[dict], runs: int) -> dict:
    by_run = [[row for row in rows if row["run"] == run] for run in range(runs)]

    def share(hit: Callable[[dict], object], pool: Callable[[dict], object]) -> dict:
        pools = [[row for row in group if pool(row)] for group in by_run]
        return over_runs([sum(bool(hit(r)) for r in p) for p in pools], [len(p) for p in pools])

    return {
        # "Keine belegte Antwort" and a refusal both count as not answered.
        "answer_rate": share(answered, lambda r: r["answerable"]),
        # Of the first replies that gave an answer, the share that passed the check as they were.
        "citation_validity": share(
            lambda r: answered(r) and not r["failures"], lambda r: answered(r) or r["failures"]
        ),
        # "partly" counts as not correct. Only shown answers are judged.
        "correctness": share(lambda r: r.get("verdict") == "correct", lambda r: r.get("verdict")),
        "faithfulness": over_runs(
            [sum(r.get("supported", 0) for r in group) for group in by_run],
            [sum(r.get("sentences", 0) for r in group) for group in by_run],
        ),
        "refusal_rate": share(lambda r: not answered(r), lambda r: not r["answerable"]),
        "refusal_rate_near_domain": share(lambda r: not answered(r), lambda r: r["near_domain"]),
    }


def cost(model: str) -> float:
    used = llm.USAGE.get(model, {})
    price_in, price_out = PRICES[model]
    return (used.get("prompt", 0) * price_in + used.get("completion", 0) * price_out) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    for model in (config.ANSWER_MODEL, config.JUDGE_MODEL):
        if model not in PRICES:
            raise SystemExit(f"Add the list price of {model} to PRICES")

    lines = (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [q for q in map(json.loads, lines) if q["split"] == args.split]
    ranked = ranked_passages(questions)

    rows = []
    for run in range(args.runs):
        for question in questions:
            result, seconds = timed_answer(question["question"], ranked[question["question"]])
            row = {
                "id": question["id"],
                "run": run,
                "answerable": question["answerable"],
                "near_domain": question.get("near_domain", False),
                "status": result.status,
                "seconds": round(seconds, 2),
                "failures": result.failures,
                "question": question["question"],
                "reference": question.get("reference_answer"),
                "answer": result.text,
                "citations": [citation.model_dump() for citation in result.citations],
                "sources": [f"{chunk['doc']} {chunk['section']}" for chunk in result.passages],
            }
            if answered(row) and question["answerable"]:
                row |= grade(question, result)
            rows.append(row)
            print(f"run {run + 1}  {row['id']}  {row.get('verdict') or row['status']}")

    DETAILS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    outcomes: dict[str, list[str]] = {}
    for row in rows:
        outcomes.setdefault(row["id"], []).append(row.get("verdict") or row["status"])
    times = sorted(row["seconds"] for row in rows)
    git = ["git", "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, cwd=config.ROOT, capture_output=True, text=True, check=False)
    result = {
        "date": date.today().isoformat(),
        "commit": commit.stdout.strip() or "uncommitted",
        "split": args.split,
        "runs": args.runs,
        "questions": len(questions),
        "origins": sorted({q["origin"] for q in questions}),
        "documents": len(config.API_DOCS),
        "answer_model": config.ANSWER_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "judge_effort": config.JUDGE_EFFORT,
        "prompts_sha256": hashlib.sha256((prompts.SYSTEM + JUDGE_TEXT).encode()).hexdigest()[:12],
        "metrics": metrics(rows, args.runs),
        # The API only. Search and reranking are timed on the Space, not on this CPU.
        "api_seconds": {
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[-1] if len(times) > 1 else times[0],
        },
        "usd_per_question": round(cost(config.ANSWER_MODEL) / len(rows), 5),
        "usd_judge_total": round(cost(config.JUDGE_MODEL), 3),
        "tokens": {model: dict(used) for model, used in llm.USAGE.items()},
        "outcomes": outcomes,
    }
    for name, value in result["metrics"].items():
        print(f"{name:26} {json.dumps(value)}")
    print(f"api_seconds {result['api_seconds']}  usd_per_question {result['usd_per_question']}")
    path = RESULTS / f"{result['date']}_{result['commit']}_answers.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(config.ROOT)}, details in {DETAILS.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()

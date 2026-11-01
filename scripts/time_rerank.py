"""Week 1 timing: how long does reranking real-length passages take on this machine?

--threads 2 is the spec's rough stand-in for CPU Basic. Each question is scored against
its known answer plus random chunks, the "typical" set, and against the longest chunks,
the "long" set, where every pair costs the most. The answer's rank is a sanity check of
the model's German, not a retrieval metric: random chunks are easy to beat.
"""

import argparse
import json
import platform
import random
import statistics
import time
from datetime import date
from pathlib import Path

import torch

from lotse.chunking import passage
from lotse.config import CHUNKS, RERANK_CANDIDATES, RERANKER_MODEL, ROOT
from lotse.retrieval.rerank import load, score

RESULTS = ROOT / "eval" / "results" / "rerank_timing.jsonl"

# Question in an employee's words, document, start of the section that answers it.
QUESTIONS = [
    ("Wie warm muss es im Büro sein?", "ArbStättV", "Anhang Nr. 3.5"),
    ("Wie lange darf ich pro Tag höchstens arbeiten?", "ArbZG", "§ 3"),
    ("Wie viele Pausen stehen mir bei einem langen Arbeitstag zu?", "ArbZG", "§ 4"),
    ("Wer ist für die Gefährdungsbeurteilung verantwortlich?", "ArbSchG", "§ 5"),
    ("Wie oft muss ich meine Leute unterweisen?", "ArbStättV", "§ 6"),
    ("Was muss in einer Betriebsanweisung für Gefahrstoffe stehen?", "GefStoffV", "§ 14 Abs. 1"),
    ("Ab wann müssen wir Gehörschutz zur Verfügung stellen?", "LärmVibrationsArbSchV", "§ 8"),
    (
        "Wie oft müssen Druckbehälter von Feuerlöschern geprüft werden?",
        "BetrSichV",
        "Anhang 2 Abschnitt 4 Tabelle 12 Ziffer 7.10",
    ),
]


def milliseconds(question: str, passages: list[str], model: str) -> tuple[float, list[float]]:
    start = time.perf_counter()
    scores = score(question, passages, model)
    return (time.perf_counter() - start) * 1000, scores


def timing_sets(candidates: int) -> dict:
    """The longest chunks, plus each question with its answer first and random chunks after."""
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    rng = random.Random(0)
    longest = sorted(chunks, key=lambda c: len(c["text"]))[-candidates:]
    typical = []
    for question, doc, section in QUESTIONS:
        answers = [c for c in chunks if c["doc"] == doc and c["section"].startswith(section)]
        others = rng.sample([c for c in chunks if c not in answers], candidates - 1)
        typical.append(
            {"question": question, "passages": [passage(c) for c in [answers[0], *others]]}
        )
    return {"long": [passage(chunk) for chunk in longest], "typical": typical}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=RERANKER_MODEL)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--export", help="write the passages to this JSON file and stop")
    args = parser.parse_args()

    sets = timing_sets(args.candidates)
    if args.export:
        Path(args.export).write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")
        return

    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    load(args.model)
    load_seconds = time.perf_counter() - start

    milliseconds(QUESTIONS[0][0], sets["long"], args.model)  # the first call pays one-off setup

    typical, long, ranks = [], [], []
    for item in sets["typical"]:
        elapsed, scores = milliseconds(item["question"], item["passages"], args.model)
        typical.append(elapsed)
        ranks.append(1 + sum(s > scores[0] for s in scores[1:]))
        long.append(milliseconds(item["question"], sets["long"], args.model)[0])

    result = {
        "date": date.today().isoformat(),
        "machine": platform.processor() or platform.machine(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": args.model,
        "threads": args.threads,
        "candidates": args.candidates,
        "load_s": round(load_seconds, 1),
        "typical_ms_p50": round(statistics.median(typical)),
        "typical_ms_max": round(max(typical)),
        "long_ms_p50": round(statistics.median(long)),
        "long_ms_max": round(max(long)),
        "answer_ranks": ranks,
    }
    print(json.dumps(result, ensure_ascii=False))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

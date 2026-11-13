"""One question in, one checked answer out.

uv run --env-file .env python -m lotse.pipeline "Wie lange darf ich pro Tag arbeiten?"
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from lotse.chunking import passage
from lotse.config import ANSWER_PASSAGES, API_DOCS, REFUSAL_THRESHOLD
from lotse.generation import llm, prompts
from lotse.generation.answer import NO_ANSWER, UNVERIFIED, Answer, Citation, check
from lotse.retrieval import dense, hybrid, rerank
from lotse.retrieval.index import Index, load_index

Ranked = list[tuple[float, dict]]


@dataclass(frozen=True)
class Result:
    status: str  # "answered", "no_answer" or "unverified"
    text: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)  # in [n] order
    failures: list[str] = field(default_factory=list)  # why the check rejected an attempt


def retrieve(index: Index, question: str) -> Ranked:
    """The best passages with their rerank scores, best first, from API_DOCS only."""
    vector = dense.embed([question])[0]
    chunks = [index.chunks[p] for p in hybrid.search(index, question, vector, API_DOCS)]
    scores = rerank.score(question, [passage(chunk) for chunk in chunks])
    return sorted(zip(scores, chunks), key=lambda pair: -pair[0])[:ANSWER_PASSAGES]


def answer(
    question: str,
    ranked: Ranked,
    complete: Callable[[list[dict]], Answer | None] = llm.complete,
    threshold: float | None = REFUSAL_THRESHOLD,
) -> Result:
    if not ranked or (threshold is not None and ranked[0][0] < threshold):
        return Result("no_answer", NO_ANSWER)  # off-topic, so the API is never called
    passages = [chunk for _, chunk in ranked]
    messages = prompts.build(question, passages)
    failures = []
    for _ in range(2):  # the second try is told what was wrong with the first
        reply = complete(messages)
        if reply and not reply.answerable:
            return Result("no_answer", NO_ANSWER, passages=passages, failures=failures)
        reason = check(reply, passages, question) if reply else prompts.INVALID
        if reason is None:
            return Result("answered", reply.answer, reply.citations, passages, failures)
        failures.append(reason)
        messages = prompts.retry(messages, reply, reason)
    return Result("unverified", UNVERIFIED, passages=passages, failures=failures)


@cache
def index() -> Index:
    return load_index()


def ask(question: str) -> Result:
    return answer(question, retrieve(index(), question))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--show-prompt", action="store_true", help="print what would be sent")
    args = parser.parse_args()

    ranked = retrieve(index(), args.question)
    for n, (score, chunk) in enumerate(ranked, start=1):
        print(f"[{n}] {score:6.2f}  {chunk['doc']} {chunk['section']}  {chunk['url']}")
    if args.show_prompt:
        messages = prompts.build(args.question, [chunk for _, chunk in ranked])
        print("\n\n".join(message["content"] for message in messages))
        return
    result = answer(args.question, ranked)
    print(f"\n{result.status}: {result.text}")
    for citation in result.citations:
        print(f"  [{citation.source}] „{citation.quote}“")
    for failure in result.failures:
        print(f"  rejected: {failure}")


if __name__ == "__main__":
    main()

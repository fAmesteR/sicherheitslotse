"""The only file that knows the API provider."""

import os
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

from mistralai.client import Mistral
from mistralai.client.utils import BackoffStrategy, RetryConfig
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from lotse.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL,
    ANSWER_TIMEOUT_MS,
    JUDGE_EFFORT,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TIMEOUT_MS,
    PRICES,
)
from lotse.generation.answer import Answer

# Tokens used per model since the program started, for the cost of an evaluation run.
USAGE: dict[str, Counter] = {}
USAGE_LOCK = threading.Lock()
# What the calls inside one meter() cost. Each thread has its own, so two questions answered at
# the same time don't mix their costs.
SPENT: ContextVar[Counter | None] = ContextVar("spent", default=None)
# Rate limits and server errors are retried with backoff for up to 15 s, then the call fails.
RETRIES = RetryConfig(
    "backoff",
    BackoffStrategy(initial_interval=500, max_interval=4000, exponent=2, max_elapsed_time=15_000),
    retry_connection_errors=True,
    status_codes_override=["429", "5XX"],
)


def usd(model: str, prompt: int, completion: int) -> float:
    """At list prices."""
    return (prompt * PRICES[model][0] + completion * PRICES[model][1]) / 1_000_000


def totals() -> tuple[int, int, float]:
    """Prompt tokens, completion tokens and USD, since the program started."""
    with USAGE_LOCK:
        used = {model: Counter(counts) for model, counts in USAGE.items()}
    prompt = sum(counts["prompt"] for counts in used.values())
    completion = sum(counts["completion"] for counts in used.values())
    return prompt, completion, sum(usd(m, c["prompt"], c["completion"]) for m, c in used.items())


def charge(model: str, prompt: int, completion: int) -> None:
    """Add one call's tokens to the totals and to the meter that is open in this thread."""
    with USAGE_LOCK:
        USAGE.setdefault(model, Counter()).update(prompt=prompt, completion=completion)
    if (spent := SPENT.get()) is not None:
        spent.update(prompt=prompt, completion=completion, usd=usd(model, prompt, completion))


@contextmanager
def meter() -> Iterator[Counter]:
    """Counts prompt tokens, completion tokens and USD of the calls made inside it."""
    spent: Counter = Counter()
    token = SPENT.set(spent)
    try:
        yield spent
    finally:
        SPENT.reset(token)


@cache
def client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Put it in .env and use: uv run --env-file .env"
        )
    return Mistral(api_key=key, retry_config=RETRIES)


def ask[Reply: BaseModel](
    model: str,
    messages: list[dict],
    schema: type[Reply],
    max_tokens: int,
    effort: str,
    timeout_ms: int,
) -> Reply | None:
    """One reply in the schema. None if it was cut off or doesn't fit the schema."""
    response = client().chat.complete(
        model=model,
        messages=messages,
        response_format=response_format_from_pydantic_model(schema),  # schema mode, strict
        temperature=0,  # still not repeatable: the same request gave different replies
        top_p=1,  # the large model demands this with temperature 0
        max_tokens=max_tokens,
        reasoning_effort=effort,
        timeout_ms=timeout_ms,  # without it the SDK waits up to five minutes
    )
    used = response.usage
    charge(model, used.prompt_tokens or 0, used.completion_tokens or 0)
    content = response.choices[0].message.content
    if not isinstance(content, str):  # a reasoning model replies in chunks, its thoughts first
        content = "".join(getattr(chunk, "text", "") for chunk in content or [])
    try:
        return schema.model_validate_json(content)
    except ValueError:  # a reply cut off at max_tokens is not valid JSON
        return None


def complete(messages: list[dict]) -> Answer | None:
    return ask(ANSWER_MODEL, messages, Answer, ANSWER_MAX_TOKENS, "none", ANSWER_TIMEOUT_MS)


def judge[Reply: BaseModel](messages: list[dict], schema: type[Reply]) -> Reply | None:
    """The same kind of call to a second model, which grades what the first one wrote."""
    return ask(JUDGE_MODEL, messages, schema, JUDGE_MAX_TOKENS, JUDGE_EFFORT, JUDGE_TIMEOUT_MS)

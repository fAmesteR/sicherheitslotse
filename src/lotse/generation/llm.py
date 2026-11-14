"""The only file that knows the API provider."""

import os
from collections import Counter
from functools import cache

from mistralai.client import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel

from lotse.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL,
    JUDGE_EFFORT,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    PRICES,
)
from lotse.generation.answer import Answer

# Tokens used per model since the program started, for the cost per answer.
USAGE: dict[str, Counter] = {}


def totals() -> tuple[int, int, float]:
    """Prompt tokens, completion tokens and USD at list prices, since the program started."""
    prompt = sum(used["prompt"] for used in USAGE.values())
    completion = sum(used["completion"] for used in USAGE.values())
    usd = sum(
        (used["prompt"] * PRICES[model][0] + used["completion"] * PRICES[model][1]) / 1_000_000
        for model, used in USAGE.items()
    )
    return prompt, completion, usd


@cache
def client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Put it in .env and use: uv run --env-file .env"
        )
    return Mistral(api_key=key)


def ask[Reply: BaseModel](
    model: str, messages: list[dict], schema: type[Reply], max_tokens: int, effort: str
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
    )
    used = response.usage
    USAGE.setdefault(model, Counter()).update(
        prompt=used.prompt_tokens or 0, completion=used.completion_tokens or 0
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):  # a reasoning model replies in chunks, its thoughts first
        content = "".join(getattr(chunk, "text", "") for chunk in content or [])
    try:
        return schema.model_validate_json(content)
    except ValueError:  # a reply cut off at max_tokens is not valid JSON
        return None


def complete(messages: list[dict]) -> Answer | None:
    return ask(ANSWER_MODEL, messages, Answer, ANSWER_MAX_TOKENS, "none")


def judge[Reply: BaseModel](messages: list[dict], schema: type[Reply]) -> Reply | None:
    """The same kind of call to a second model, which grades what the first one wrote."""
    return ask(JUDGE_MODEL, messages, schema, JUDGE_MAX_TOKENS, JUDGE_EFFORT)

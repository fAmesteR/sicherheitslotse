"""The only file that knows the API provider."""

import os
from functools import cache

from mistralai.client import Mistral
from mistralai.extra import response_format_from_pydantic_model

from lotse.config import ANSWER_MAX_TOKENS, ANSWER_MODEL
from lotse.generation.answer import Answer


@cache
def client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Put it in .env and use: uv run --env-file .env"
        )
    return Mistral(api_key=key)


def complete(messages: list[dict]) -> Answer | None:
    """The model's reply in the schema. None if it was cut off or doesn't fit the schema."""
    response = client().chat.complete(
        model=ANSWER_MODEL,
        messages=messages,
        response_format=response_format_from_pydantic_model(Answer),  # schema mode, strict
        temperature=0,  # still not repeatable: the same request gave different replies
        max_tokens=ANSWER_MAX_TOKENS,
        reasoning_effort="none",
    )
    content = response.choices[0].message.content
    try:
        return Answer.model_validate_json(content) if isinstance(content, str) else None
    except ValueError:  # a reply cut off at max_tokens is not valid JSON
        return None

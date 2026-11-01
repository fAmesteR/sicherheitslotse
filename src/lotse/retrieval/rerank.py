"""Score passages against a question with a cross-encoder."""

from functools import cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lotse.config import RERANK_MAX_LENGTH, RERANKER_MODEL


@cache
def load(name: str = RERANKER_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval()


def score(question: str, passages: list[str], name: str = RERANKER_MODEL) -> list[float]:
    """One relevance score per passage, higher is better. The scale depends on the model."""
    tokenizer, model = load(name)
    batch = tokenizer(
        [[question, passage] for passage in passages],
        padding=True,
        truncation=True,
        max_length=RERANK_MAX_LENGTH,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        return model(**batch).logits.view(-1).tolist()

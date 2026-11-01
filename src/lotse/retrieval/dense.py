"""Meaning search: bge-m3 embeddings compared by cosine similarity."""

from functools import cache

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from lotse.config import EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL


@cache
def load():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval()


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """One L2-normalised CLS vector per text, in input order.

    Texts are batched by length, because a batch pads to its longest text.
    """
    tokenizer, model = load()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vectors = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        batch = tokenizer(
            [texts[i] for i in positions],
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            cls = model(**batch).last_hidden_state[:, 0]
        vectors[positions] = torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
    return vectors


def search(embeddings: np.ndarray, question: np.ndarray, k: int) -> list[int]:
    """Chunk positions, best first. For normalised vectors the dot product is cosine similarity."""
    return np.argsort(-(embeddings @ question))[:k].tolist()

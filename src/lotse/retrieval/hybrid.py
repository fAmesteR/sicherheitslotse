"""Merge keyword search and meaning search with Reciprocal Rank Fusion."""

import numpy as np

from lotse.config import RERANK_CANDIDATES, RRF_K, SEARCH_CANDIDATES
from lotse.retrieval import bm25, dense
from lotse.retrieval.index import Index


def reciprocal_rank_fusion(
    rankings: list[list[int]], limit: int = RERANK_CANDIDATES, k: int = RRF_K
) -> list[int]:
    """Best items across the rankings by the sum of 1 / (k + rank). Ties keep first-seen order."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:limit]


def search(
    index: Index, question: str, vector: np.ndarray, docs: frozenset[str] | None = None
) -> list[int]:
    """Chunk positions to rerank. The vector comes from dense.embed, which needs the GPU.

    With docs, only chunks of those documents come back, and each search still supplies up to
    SEARCH_CANDIDATES of them.
    """

    def candidates(positions: list[int]) -> list[int]:
        allowed = [p for p in positions if docs is None or index.chunks[p]["doc"] in docs]
        return allowed[:SEARCH_CANDIDATES]

    k = SEARCH_CANDIDATES if docs is None else len(index.chunks)
    return reciprocal_rank_fusion(
        [
            candidates(bm25.search(index.bm25, question, k)),
            candidates(dense.search(index.embeddings, vector, k)),
        ]
    )

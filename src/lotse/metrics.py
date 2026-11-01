"""Retrieval metrics. `gold` lists what an answer needs; each inner list can supply one part."""

from math import sqrt

Gold = list[list[str]]


def in_section(chunk: dict, label: str) -> bool:
    """True if the chunk is the labelled section or sits under it, so re-chunking keeps labels."""
    name = f"{chunk['doc']} {chunk['section']}"
    return name == label or name.startswith((f"{label} ", f"{label}."))


def has_all_gold(chunks: list[dict], gold: Gold) -> bool:
    return all(any(in_section(c, label) for c in chunks for label in part) for part in gold)


def reciprocal_rank(chunks: list[dict], gold: Gold) -> float:
    """1 / rank of the first chunk from a gold section, 0 if there is none."""
    labels = [label for part in gold for label in part]
    for rank, chunk in enumerate(chunks, start=1):
        if any(in_section(chunk, label) for label in labels):
            return 1 / rank
    return 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share. Unlike the normal one, it stays sensible on small test sets."""
    if n == 0:
        return 0.0, 1.0
    share = successes / n
    centre = share + z * z / (2 * n)
    spread = z * sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (centre - spread) / denominator, (centre + spread) / denominator

"""Retrieval metrics. `gold` lists what an answer needs; each inner list can supply one part."""

from math import ceil, sqrt
from statistics import fmean

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


def over_runs(hits: list[int], totals: list[int]) -> dict:
    """A share measured in several runs of the same questions. The interval is for an average
    run and covers the choice of questions. per_run shows the model's own run-to-run noise."""
    shares = [hit / total for hit, total in zip(hits, totals) if total]
    if not shares:
        return {"share": None, "n": 0}
    n = round(fmean(totals))
    low, high = wilson_interval(round(fmean(hits)), n)
    return {
        "share": round(fmean(shares), 3),
        "per_run": [round(share, 3) for share in shares],
        "n": n,
        "interval_95": [round(low, 3), round(high, 3)],
    }


def regressions(current: dict, baseline: dict, tolerance: float = 0.02) -> list[str]:
    """What fell more than `tolerance` below the stored baseline, one line per metric. Small
    changes move the numbers a little by chance, so a gate without tolerance would cry wolf."""
    return [
        f"{name}: {current[name]:.3f} is more than {tolerance} below the baseline {expected:.3f}"
        for name, expected in baseline["metrics"].items()
        if current[name] < expected - tolerance
    ]


def percentile(ordered: list[float], share: float) -> float | None:
    """The value that `share` of an ascending list lies at or below (nearest rank)."""
    if not ordered:
        return None
    return ordered[max(0, ceil(share * len(ordered)) - 1)]

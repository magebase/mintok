"""Simple descriptive statistics over numeric sequences."""

from __future__ import annotations

from biglib.numbers import clamp


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("variance needs at least two values")
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def summarize(values: list[float], precision: int = 2) -> dict[str, float]:
    """Headline stats; precision is clamped to 0..6 decimal places."""
    digits = clamp(precision, 0, 6)
    return {
        "mean": round(mean(values), digits),
        "median": round(median(values), digits),
        "variance": round(variance(values), digits),
    }

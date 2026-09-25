"""Efficiency accounting: accepted changes per total dollar, with paired statistics."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RunRecord:
    task_id: str
    arm: str
    solved: bool
    frontier_usd: float
    local_usd: float = 0.0
    indexing_usd: float = 0.0
    storage_usd: float = 0.0
    cpu_usd: float = 0.0
    frontier_calls: int = 0
    turns: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_usd(self) -> float:
        return self.frontier_usd + self.local_usd + self.indexing_usd + self.storage_usd + self.cpu_usd


def _arm(records: Iterable[RunRecord], arm: str) -> list[RunRecord]:
    return [r for r in records if r.arm == arm]


def accepted_per_dollar(records: Iterable[RunRecord]) -> float:
    records = list(records)
    cost = sum(r.total_usd for r in records)
    solved = sum(r.solved for r in records)
    return solved / cost if cost > 0 else float("inf")


def frontier_calls_per_success(records: Iterable[RunRecord]) -> float:
    records = list(records)
    solved = sum(r.solved for r in records)
    return sum(r.frontier_calls for r in records) / solved if solved else float("inf")


def efficiency_ratio(records: Iterable[RunRecord], treatment: str, baseline: str) -> float:
    records = list(records)
    return accepted_per_dollar(_arm(records, treatment)) / accepted_per_dollar(_arm(records, baseline))


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float

    @property
    def significant(self) -> bool:
        return self.lower > 1.0 or self.upper < 1.0


def paired_bootstrap_ratio(
    records: Iterable[RunRecord],
    treatment: str,
    baseline: str,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample tasks (not runs) so repeated runs of one task stay paired."""
    per_task: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {treatment: [0.0, 0.0], baseline: [0.0, 0.0]}
    )
    for r in records:
        if r.arm in (treatment, baseline):
            cell = per_task[r.task_id][r.arm]
            cell[0] += r.solved
            cell[1] += r.total_usd

    tasks = [t for t, arms in per_task.items() if arms[treatment][1] > 0 and arms[baseline][1] > 0]
    if not tasks:
        raise ValueError("no paired tasks with cost in both arms")

    def ratio(sample: list[str]) -> float | None:
        ts = sum(per_task[t][treatment][0] for t in sample)
        tc = sum(per_task[t][treatment][1] for t in sample)
        bs = sum(per_task[t][baseline][0] for t in sample)
        bc = sum(per_task[t][baseline][1] for t in sample)
        if bs == 0:
            return None
        return (ts / tc) / (bs / bc)

    point = ratio(tasks)
    if point is None:
        raise ValueError("baseline solved no tasks; ratio undefined")

    rng = random.Random(seed)
    stats = sorted(
        v for v in (ratio([rng.choice(tasks) for _ in tasks]) for _ in range(resamples)) if v is not None
    )
    lo = stats[int((alpha / 2) * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return BootstrapResult(point, lo, hi)


def oracle_verdict(headroom: float, kill_below: float = 2.0, continue_at: float = 3.0) -> str:
    """Oracle-context headroom gates investment in context architecture."""
    if headroom < kill_below:
        return "KILL"
    if headroom < continue_at:
        return "INVESTIGATE"
    return "CONTINUE"

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PricingTable:
    """Externalized, versioned provider pricing rate card.

    Raw provider token usage never changes; pricing data is tracked externally
    with content hash to allow historical cost re-evaluation as model prices evolve.
    """

    provider: str = "openrouter"
    model: str = "anthropic/claude-3.5-sonnet"
    effective_from: str = "2026-09-01"
    fresh_input_price: float = 3.00
    cache_read_price: float = 0.30
    cache_write_price: float = 3.75
    output_price: float = 15.00
    reasoning_price: float = 15.00

    def pricing_table_hash(self) -> str:
        payload = (
            f"{self.provider}:{self.model}:{self.effective_from}:"
            f"{self.fresh_input_price}:{self.cache_read_price}:"
            f"{self.cache_write_price}:{self.output_price}:{self.reasoning_price}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Backwards compatibility alias
ProviderPricing = PricingTable


@dataclass(frozen=True, slots=True)
class TokenUsageBreakdown:
    """Four independent counters for actual provider cost calculation."""

    t_fresh: int = 0
    t_cached: int = 0
    t_output: int = 0
    t_reasoning: int = 0
    pricing_table_hash: str = ""

    @property
    def total_tokens(self) -> int:
        return self.t_fresh + self.t_cached + self.t_output + self.t_reasoning

    def compute_billed_usd(self, pricing: PricingTable | None = None) -> float:
        p = pricing or PricingTable()
        fresh_usd = (self.t_fresh / 1_000_000.0) * p.fresh_input_price
        cached_usd = (self.t_cached / 1_000_000.0) * p.cache_read_price
        output_usd = (self.t_output / 1_000_000.0) * p.output_price
        reasoning_usd = (self.t_reasoning / 1_000_000.0) * p.reasoning_price
        return fresh_usd + cached_usd + output_usd + reasoning_usd


def frontier_call_elimination_ratio(frontier_calls_mintok: int, frontier_calls_control: int) -> float:
    """FER = 1 - (frontier_calls_mintok / frontier_calls_control)."""
    if frontier_calls_control <= 0:
        return 0.0
    return max(0.0, 1.0 - (float(frontier_calls_mintok) / float(frontier_calls_control)))


def decisive_evidence_density(decisive_tokens: int, search_tokens_before: int) -> float:
    """DED = decisive_evidence_tokens / tokens_before_decisive_evidence."""
    return float(decisive_tokens) / float(max(1, search_tokens_before))


def post_decisive_efficiency(tokens_after_decisive: int, total_tokens: int) -> float:
    """PDE = tokens_after_decisive / total_tokens."""
    return float(tokens_after_decisive) / float(max(1, total_tokens))


def information_reuse_multiplier(uses_count: int, creation_tokens: int) -> float:
    """IRM = uses_of_stored_knowledge / tokens_spent_creating_knowledge."""
    return float(uses_count) / float(max(1, creation_tokens))


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
    usage_breakdown: TokenUsageBreakdown | None = None

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


# ---------------------------------------------------------------------------
# Oracle router: per-task best-arm selection as an offline upper bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingReport:
    """What a perfect per-task arm selector would have achieved."""

    tasks: int
    solved: int
    tokens_per_attempt: float
    tokens_per_solved: float
    solves_per_mtok: float
    p50: int
    p95: int
    max_tokens: int
    routed_to: dict[str, str]

    def uniform(self, records: Iterable[RunRecord], arm: str) -> float:
        """Tokens per attempted task if the whole workload ran on ``arm``."""
        rows = [r for r in records if r.arm == arm]
        total = sum(r.input_tokens for r in rows)
        return total / len(rows) if rows else 0.0


def oracle_router(runs: Iterable[RunRecord], arm_a: str, arm_b: str) -> RoutingReport:
    """Pick, per task, the arm a perfect router would send the task to.

    A task both arms solve goes to the cheaper solver; a task only one arm
    solves goes to the solver; a task neither solves is charged the cheaper
    failed attempt (the optimistic bound — a real router cannot know in
    advance). Works on the latest record per (task, arm) so blind-append
    record files stay usable.
    """
    latest: dict[tuple[str, str], RunRecord] = {}
    for run in runs:
        if run.arm in (arm_a, arm_b):
            latest[(run.task_id, run.arm)] = run

    task_ids = sorted({task for task, _ in latest})
    routed: dict[str, RunRecord] = {}
    for task_id in task_ids:
        a = latest.get((task_id, arm_a))
        b = latest.get((task_id, arm_b))
        if a is None or b is None:
            chosen = a or b
        elif a.solved and b.solved:
            chosen = a if a.input_tokens <= b.input_tokens else b
        elif a.solved:
            chosen = a
        elif b.solved:
            chosen = b
        else:
            chosen = a if a.input_tokens <= b.input_tokens else b
        routed[task_id] = chosen

    rows = list(routed.values())
    solved = sum(r.solved for r in rows)
    total = sum(r.input_tokens for r in rows)
    tokens = sorted(r.input_tokens for r in rows)

    def pct(p: float) -> int:
        return tokens[min(len(tokens) - 1, math.ceil(p * len(tokens)) - 1)]

    return RoutingReport(
        tasks=len(rows),
        solved=solved,
        tokens_per_attempt=total / len(rows) if rows else 0.0,
        tokens_per_solved=total / solved if solved else float("inf"),
        solves_per_mtok=(solved / total * 1_000_000) if total else 0.0,
        p50=tokens[len(tokens) // 2] if tokens else 0,
        p95=pct(0.95) if tokens else 0,
        max_tokens=max(tokens) if tokens else 0,
        routed_to={task: run.arm for task, run in routed.items()},
    )


# Cost-accounting spec: $0.01 of blended cost per Optimization Compute Unit.
# A pure, published transformation of measured cost — never a token markup.
OCU_USD = 0.01


def optimization_compute_units(total_usd: float) -> float:
    """Meter blended cost into Optimization Compute Units (OCU)."""
    return total_usd / OCU_USD

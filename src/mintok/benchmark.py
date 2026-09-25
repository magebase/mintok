"""Agent Efficiency Benchmark: paired arms, cache-aware $/solved, raw traces.

The runner, statistics, and trace formats are open so efficiency claims are
independently verifiable: anyone can check the API bills and patches without
seeing how the optimizer chose its context. The optimizer under test is
pluggable; the reference optimizer is the commercial MinTok Inference Compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from mintok.metrics import RunRecord
from mintok.records import dump_runs_jsonl

Runner = Callable[[str], dict]


@dataclass(frozen=True, slots=True)
class ArmStats:
    arm: str
    solved: int
    total: int
    cost: float

    @property
    def solved_rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def cost_per_solved(self) -> float:
        return self.cost / self.solved if self.solved else float("inf")


def arm_stats(records: Iterable[RunRecord], arm: str) -> ArmStats:
    rows = [r for r in records if r.arm == arm]
    return ArmStats(
        arm=arm,
        solved=sum(r.solved for r in rows),
        total=len(rows),
        cost=sum(r.total_usd for r in rows),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    baseline: ArmStats
    optimizer: ArmStats

    @property
    def work_multiple(self) -> float:
        return self.baseline.cost_per_solved / self.optimizer.cost_per_solved

    @property
    def success_regression(self) -> bool:
        return self.optimizer.solved_rate < self.baseline.solved_rate

    def render(self) -> str:
        lines = [
            "Agent Efficiency Benchmark",
            f"{'arm':<11}{'solved':>8}{'$/solve':>10}{'work/$':>9}",
            f"{'baseline':<11}{f'{self.baseline.solved}/{self.baseline.total}':>8}"
            f"{f'${self.baseline.cost_per_solved:.2f}':>10}{'1.00x':>9}",
            f"{'optimizer':<11}{f'{self.optimizer.solved}/{self.optimizer.total}':>8}"
            f"{f'${self.optimizer.cost_per_solved:.2f}':>10}{f'{self.work_multiple:.2f}x':>9}",
        ]
        lines.append("success regression: YES" if self.success_regression else "success regression: none")
        return "\n".join(lines)


def benchmark_report(records: Iterable[RunRecord]) -> BenchmarkReport:
    records = list(records)
    return BenchmarkReport(baseline=arm_stats(records, "baseline"), optimizer=arm_stats(records, "optimizer"))


@dataclass(frozen=True, slots=True)
class SavingsSummary:
    tasks: int
    baseline_usd: float
    optimizer_usd: float
    solved: int

    @property
    def saved(self) -> float:
        return self.baseline_usd - self.optimizer_usd

    @property
    def reduction_pct(self) -> float:
        return self.saved / self.baseline_usd * 100 if self.baseline_usd else 0.0

    @property
    def multiple(self) -> float:
        return self.baseline_usd / self.optimizer_usd if self.optimizer_usd else float("inf")

    def render(self) -> str:
        return "\n".join(
            [
                "Savings summary",
                f"tasks optimized     {self.tasks:>6}",
                f"baseline estimate  ${self.baseline_usd:>8,.2f}",
                f"actual spend       ${self.optimizer_usd:>8,.2f}",
                f"saved              ${self.saved:>8,.2f}",
                f"reduction           {self.reduction_pct:>6.1f}%",
                f"work/$             {self.multiple:>7.2f}x",
            ]
        )


def savings_summary(records: Iterable[RunRecord]) -> SavingsSummary:
    baseline = arm_stats(records, "baseline")
    optimizer = arm_stats(records, "optimizer")
    return SavingsSummary(
        tasks=baseline.total,
        baseline_usd=baseline.cost,
        optimizer_usd=optimizer.cost,
        solved=min(baseline.solved, optimizer.solved),
    )


def run_benchmark(
    tasks: list[str],
    runners: dict[str, Runner],
    out_dir: str | Path,
) -> dict[str, list[RunRecord]]:
    """Run every task under every arm and write one raw-trace JSONL per arm.

    A runner receives a task id and returns solved/usd/tokens; wiring real
    frontier agents into it happens outside the open test suite.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[RunRecord]] = {}
    for arm, runner in runners.items():
        records = [
            RunRecord(
                task_id=task,
                arm=arm,
                solved=bool(row["solved"]),
                frontier_usd=float(row["usd"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
            )
            for task in tasks
            for row in (runner(task),)
        ]
        dump_runs_jsonl(records, out / f"{arm}.jsonl")
        results[arm] = records
    return results

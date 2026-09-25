from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.benchmark import benchmark_report, run_benchmark, savings_summary
from mintok.metrics import RunRecord
from mintok.records import dump_runs_jsonl
from tests.acceptance.helpers import table_rows

scenarios("benchmark.feature")


@given("these paired benchmark results:")
def paired_results(ctx: SimpleNamespace, datatable: list[list[str]]) -> None:
    ctx.records = []
    for row in table_rows(datatable):
        ctx.records.append(
            RunRecord(
                task_id=row["task"],
                arm="baseline",
                solved=row["baseline_solved"] == "yes",
                frontier_usd=float(row["baseline_usd"]),
            )
        )
        ctx.records.append(
            RunRecord(
                task_id=row["task"],
                arm="optimizer",
                solved=row["optimizer_solved"] == "yes",
                frontier_usd=float(row["optimizer_usd"]),
            )
        )


@when("the benchmark report is computed")
def compute_report(ctx: SimpleNamespace) -> None:
    ctx.report = benchmark_report(ctx.records)


@then(parsers.parse("{arm} solves {solved:d} of {total:d} at ${cost:g} per solved task"))
def arm_solves(ctx: SimpleNamespace, arm: str, solved: int, total: int, cost: float) -> None:
    stats = getattr(ctx.report, arm)
    assert (stats.solved, stats.total) == (solved, total)
    assert round(stats.cost_per_solved, 2) == cost


@then(parsers.parse("the work-per-dollar multiple is {multiple:g}"))
def work_multiple(ctx: SimpleNamespace, multiple: float) -> None:
    assert round(ctx.report.work_multiple, 2) == multiple


@then("the report does not flag a success regression")
def no_regression(ctx: SimpleNamespace) -> None:
    assert ctx.report.success_regression is False


@then("the report flags a success regression")
def regression(ctx: SimpleNamespace) -> None:
    assert ctx.report.success_regression is True


@given(
    parsers.parse(
        "the baseline arm spent ${b_usd:g} and the optimizer arm spent ${o_usd:g},"
        " both solving {solved:d} of {tasks:d} tasks"
    )
)
def savings_setup(ctx: SimpleNamespace, b_usd: float, o_usd: float, solved: int, tasks: int) -> None:
    ctx.records = []
    for i in range(1, tasks + 1):
        solved_i = i <= solved
        ctx.records.append(RunRecord(task_id=f"t{i}", arm="baseline", solved=solved_i, frontier_usd=b_usd / tasks))
        ctx.records.append(RunRecord(task_id=f"t{i}", arm="optimizer", solved=solved_i, frontier_usd=o_usd / tasks))


@when("the savings summary is rendered")
def render_savings(ctx: SimpleNamespace) -> None:
    ctx.savings = savings_summary(ctx.records)


@then(parsers.parse('it reports saved "${amount:g}"'))
def saved(ctx: SimpleNamespace, amount: float) -> None:
    assert round(ctx.savings.saved, 2) == amount


@then(parsers.parse('it reports reduction "{pct:g}%"'))
def reduction(ctx: SimpleNamespace, pct: float) -> None:
    assert round(ctx.savings.reduction_pct, 1) == pct


@then(parsers.parse('it reports a "{mult:g}x" work-per-dollar multiple'))
def savings_multiple(ctx: SimpleNamespace, mult: float) -> None:
    assert round(ctx.savings.multiple, 2) == mult


@given("a synthetic agent runner that solves every task")
def synthetic_runner(ctx: SimpleNamespace) -> None:
    def runner(task: str) -> dict:
        return {"solved": True, "usd": 0.10, "input_tokens": 500, "output_tokens": 50, "model": "synthetic"}

    ctx.runner = runner


@when("a 2-task benchmark runs both arms into a temporary output directory")
def run_two_task_benchmark(ctx: SimpleNamespace, tmp_path: Path) -> None:
    ctx.out_dir = tmp_path / "benchmark-results"
    ctx.results = run_benchmark(["t1", "t2"], {"baseline": ctx.runner, "optimizer": ctx.runner}, ctx.out_dir)


@then("the output directory holds a trace file per arm")
def trace_files_exist(ctx: SimpleNamespace) -> None:
    files = {p.name for p in ctx.out_dir.iterdir()}
    assert files == {"baseline.jsonl", "optimizer.jsonl"}


@then("each trace records task, tokens, cost, and solved as JSON lines")
def trace_contents(ctx: SimpleNamespace) -> None:
    for path in sorted(ctx.out_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 2
        for row in rows:
            assert {"task_id", "input_tokens", "output_tokens", "frontier_usd", "solved"} <= set(row)


@given("run-record files for a baseline and an optimizer arm")
def run_record_files(ctx: SimpleNamespace, repo: Path) -> None:
    baseline = [
        RunRecord(task_id=f"t{i}", arm="baseline", solved=True, frontier_usd=usd)
        for i, usd in ((1, 2.0), (2, 2.0), (3, 1.0))
    ]
    optimizer = [
        RunRecord(task_id=f"t{i}", arm="optimizer", solved=True, frontier_usd=usd)
        for i, usd in ((1, 0.50), (2, 0.50), (3, 0.25))
    ]
    dump_runs_jsonl(baseline, repo / "baseline.jsonl")
    dump_runs_jsonl(optimizer, repo / "optimizer.jsonl")

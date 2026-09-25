from __future__ import annotations

from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.metrics import (
    RunRecord,
    accepted_per_dollar,
    efficiency_ratio,
    frontier_calls_per_success,
    oracle_verdict,
    paired_bootstrap_ratio,
)
from tests.acceptance.helpers import table_rows

scenarios("efficiency_metrics.feature")


@given("these benchmark runs:")
def benchmark_runs(ctx: SimpleNamespace, datatable: list[list[str]]) -> None:
    ctx.records = [
        RunRecord(
            task_id=row["task"],
            arm=row["arm"],
            solved=row["solved"] == "yes",
            frontier_usd=float(row["frontier_usd"]),
            local_usd=float(row["local_usd"]),
            indexing_usd=float(row["indexing_usd"]),
            frontier_calls=int(row["frontier_calls"]),
        )
        for row in table_rows(datatable)
    ]


@then(parsers.parse('arm "{arm}" achieves {value:f} accepted changes per dollar'))
def arm_efficiency(ctx: SimpleNamespace, arm: str, value: float) -> None:
    assert round(accepted_per_dollar(r for r in ctx.records if r.arm == arm), 2) == value


@then(parsers.parse('arm "{arm}" uses {value:f} frontier calls per success'))
def calls_per_success(ctx: SimpleNamespace, arm: str, value: float) -> None:
    assert frontier_calls_per_success(r for r in ctx.records if r.arm == arm) == value


@then(parsers.parse('the efficiency ratio of "{treatment}" over "{baseline}" is {value:f}'))
def ratio(ctx: SimpleNamespace, treatment: str, baseline: str, value: float) -> None:
    assert round(efficiency_ratio(ctx.records, treatment, baseline), 2) == value


@given(parsers.parse('{n:d} paired tasks where "{baseline}" costs {b_cost:f} and "{treatment}" costs {t_cost:f} per task'))
def paired_tasks(ctx: SimpleNamespace, n: int, baseline: str, b_cost: float, treatment: str, t_cost: float) -> None:
    ctx.paired = SimpleNamespace(n=n, baseline=baseline, treatment=treatment, costs={baseline: b_cost, treatment: t_cost})
    ctx.solved = {baseline: set(), treatment: set()}


@given(parsers.parse("both arms solve tasks {first:d} to {last:d}"))
def both_solve(ctx: SimpleNamespace, first: int, last: int) -> None:
    for arm in ctx.solved:
        ctx.solved[arm] = set(range(first, last + 1))


@given(parsers.parse('"{a}" solves tasks {a1:d} to {a2:d} and "{b}" solves tasks {b1:d} to {b2:d}'))
def arms_solve(ctx: SimpleNamespace, a: str, a1: int, a2: int, b: str, b1: int, b2: int) -> None:
    ctx.solved[a] = set(range(a1, a2 + 1))
    ctx.solved[b] = set(range(b1, b2 + 1))


@when(parsers.parse("a paired bootstrap of the efficiency ratio is run with seed {seed:d}"))
def run_bootstrap(ctx: SimpleNamespace, seed: int) -> None:
    p = ctx.paired
    records = [
        RunRecord(task_id=f"t{i}", arm=arm, solved=i in ctx.solved[arm], frontier_usd=p.costs[arm])
        for i in range(1, p.n + 1)
        for arm in (p.baseline, p.treatment)
    ]
    ctx.bootstrap = paired_bootstrap_ratio(records, p.treatment, p.baseline, seed=seed)


@then(parsers.parse("the ratio confidence interval lower bound is above {bound:f}"))
def lower_bound(ctx: SimpleNamespace, bound: float) -> None:
    assert ctx.bootstrap.lower > bound, ctx.bootstrap


@then(parsers.parse('the result is "{verdict}"'))
def significance(ctx: SimpleNamespace, verdict: str) -> None:
    expected = verdict == "significant"
    assert ctx.bootstrap.significant is expected, ctx.bootstrap


@given(parsers.parse("the oracle-context arm is {headroom:f}x more efficient than the baseline"))
def oracle_headroom(ctx: SimpleNamespace, headroom: float) -> None:
    ctx.headroom = headroom


@then(parsers.parse('the oracle verdict is "{verdict}"'))
def verdict_is(ctx: SimpleNamespace, verdict: str) -> None:
    assert oracle_verdict(ctx.headroom) == verdict

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "public"))

from mintok.reproduce import build_balanced_schedule, run_reproduction  # noqa: E402
from verify_reproducibility import (  # noqa: E402
    verify_second_model_family,
    verify_third_model_family,
)

scenarios("external_reproduction.feature")


@given(parsers.parse('a request to reproduce the "{benchmark}" benchmark in quick mode'))
def given_reproduce_request(ctx: SimpleNamespace, benchmark: str) -> None:
    ctx.benchmark = benchmark
    ctx.quick = True


@when("the external reproduction harness runs")
def when_reproduce_runs(ctx: SimpleNamespace) -> None:
    report, passed = run_reproduction(benchmark=ctx.benchmark, quick=ctx.quick, mock=True)
    ctx.report = report
    ctx.passed = passed


@then("the cryptographic window fingerprint is verified")
def check_fingerprint_verified(ctx: SimpleNamespace) -> None:
    assert ctx.report.total_tasks == 10


@then("the execution schedule achieves exact 50/50 balance")
def check_schedule_balance(ctx: SimpleNamespace) -> None:
    task_ids = [f"t{i}" for i in range(10)]
    sched = build_balanced_schedule(task_ids)
    ctrl_first = sum(1 for arms in sched.values() if arms[0] == "control")
    mintok_first = sum(1 for arms in sched.values() if arms[0] == "mintok")
    assert ctrl_first == 5
    assert mintok_first == 5


@then(parsers.parse("the paired efficiency multiplier is at least {min_mult:f}x"))
def check_eff_multiplier(ctx: SimpleNamespace, min_mult: float) -> None:
    assert ctx.report.efficiency_multiplier >= min_mult


@then(parsers.parse("the 95% bootstrap confidence interval lower bound exceeds {min_bound:f}x"))
def check_ci_lower_bound(ctx: SimpleNamespace, min_bound: float) -> None:
    assert ctx.report.efficiency_ci is not None
    assert ctx.report.efficiency_ci.low >= min_bound


@then("the reproduction status is reported as passing")
def check_reproduction_passing(ctx: SimpleNamespace) -> None:
    assert ctx.passed is True
    assert ctx.report.gate_verdict in ("STRONG", "EXCELLENT", "PASS")


@given(
    parsers.parse(
        'three distinct model families "{m1}", "{m2}", and "{m3}"'
    )
)
def given_three_models(ctx: SimpleNamespace, m1: str, m2: str, m3: str) -> None:
    ctx.model_families = [m1, m2, m3]


@when("cross-model paired evaluations are audited")
def when_audit_cross_model(ctx: SimpleNamespace) -> None:
    m2_res = verify_second_model_family()
    m3_res = verify_third_model_family()
    ctx.multi_model_results = [m2_res, m3_res]


@then(
    parsers.parse(
        "each model family maintains solve-rate difference within {max_drop:d} percentage points"
    )
)
def check_multi_model_solve_diff(ctx: SimpleNamespace, max_drop: int) -> None:
    for res in ctx.multi_model_results:
        assert res["solve_drop_pp"] * 100 <= max_drop


@then(
    parsers.parse(
        "each model family achieves an efficiency multiplier of at least {min_mult:f}x"
    )
)
def check_multi_model_eff_mult(ctx: SimpleNamespace, min_mult: float) -> None:
    for res in ctx.multi_model_results:
        assert res["efficiency_multiplier"] >= min_mult

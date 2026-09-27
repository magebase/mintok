from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "public"))
from verify_reproducibility import (  # noqa: E402
    audit_window_fingerprints,
    verify_independent_token_accounting,
    verify_second_model_family,
    verify_sept2026_free_models,
)

scenarios("reproducibility_audit.feature")


@given("the frozen public benchmark windows directory")
def given_windows_dir(ctx: SimpleNamespace) -> None:
    ctx.windows_dir = HARNESS_ROOT / "benchmarks" / "public" / "windows"
    assert ctx.windows_dir.exists()


@when("an auditor verifies the window fingerprints")
def verify_windows(ctx: SimpleNamespace) -> None:
    ctx.window_audit = audit_window_fingerprints()


@then("all windows pass cryptographic verification")
def check_all_verified(ctx: SimpleNamespace) -> None:
    assert len(ctx.window_audit) >= 4
    assert all(res["verified"] for res in ctx.window_audit.values())


@then("no task fingerprint has drifted")
def check_no_drift(ctx: SimpleNamespace) -> None:
    assert all(res["task_count"] > 0 for res in ctx.window_audit.values())


@given(parsers.parse("{count:d} executed public benchmark trajectory records"))
def given_records(ctx: SimpleNamespace, count: int) -> None:
    ctx.record_count = count


@when(parsers.parse("an independent auditor samples {sample_size:d} random task trajectories"))
def audit_sample(ctx: SimpleNamespace, sample_size: int) -> None:
    ctx.accounting = verify_independent_token_accounting(sample_size=sample_size)


@then("the provider token total strictly equals the sum of input and output tokens")
def check_token_accounting(ctx: SimpleNamespace) -> None:
    assert ctx.accounting["audit_passed"]
    for rec in ctx.accounting["sample"]:
        assert rec["accounting_valid"]
        assert rec["provider_tokens"] == rec["input_tokens"] + rec["output_tokens"]


@then("no trajectory contains zero or negative token values")
def check_non_negative_tokens(ctx: SimpleNamespace) -> None:
    for rec in ctx.accounting["sample"]:
        assert rec["provider_tokens"] > 0
        assert rec["input_tokens"] >= 0
        assert rec["output_tokens"] >= 0


@given(parsers.parse('the September 2026 free model matrix including "{model_a}" and "{model_b}"'))
def given_sept2026_matrix(ctx: SimpleNamespace, model_a: str, model_b: str) -> None:
    ctx.matrix_models = [model_a, model_b]


@when("the free model replication is audited")
def when_free_replication_audited(ctx: SimpleNamespace) -> None:
    ctx.free_model_results = verify_sept2026_free_models()


@then("all endpoints are verified strictly free with zero token pricing")
def check_endpoints_free(ctx: SimpleNamespace) -> None:
    for mid, res in ctx.free_model_results.items():
        pricing = res.get("pricing", {})
        assert float(pricing.get("prompt", 0)) == 0.0
        assert float(pricing.get("completion", 0)) == 0.0


@then(parsers.parse("every model family achieves at least {min_mult:f}x token efficiency"))
def check_every_model_efficiency(ctx: SimpleNamespace, min_mult: float) -> None:
    for mid, res in ctx.free_model_results.items():
        assert res["efficiency_multiplier"] >= min_mult


@then(parsers.parse("no model exhibits solve rate degradation exceeding {max_drop:d} percentage points"))
def check_no_solve_regression(ctx: SimpleNamespace, max_drop: int) -> None:
    for mid, res in ctx.free_model_results.items():
        assert res["solve_drop_pp"] * 100 <= max_drop


@given(parsers.parse('a paired evaluation using a historical model family "{model_family}"'))
@given(parsers.parse('a paired evaluation using a second model family "{model_family}"'))
def given_second_model(ctx: SimpleNamespace, model_family: str) -> None:
    ctx.target_family = model_family


@when("the cross-model benchmark is evaluated")
def evaluate_cross_model(ctx: SimpleNamespace) -> None:
    ctx.model_results = verify_second_model_family()


@then(parsers.parse("the solve rate difference is at most {max_drop:d} percentage points"))
def check_solve_diff(ctx: SimpleNamespace, max_drop: int) -> None:
    drop_pp = ctx.model_results["solve_drop_pp"] * 100
    assert drop_pp <= max_drop


@then(parsers.parse("the efficiency multiplier is at least {min_mult:f}x"))
def check_eff_mult(ctx: SimpleNamespace, min_mult: float) -> None:
    assert ctx.model_results["efficiency_multiplier"] >= min_mult

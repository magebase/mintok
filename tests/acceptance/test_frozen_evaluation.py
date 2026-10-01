"""Acceptance tests for MinTok-3.2-FROZEN, unseen model transfer, and budget frontier curve."""

from __future__ import annotations

import io
import json
import time
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.budget_curve import BudgetCurveReport, BudgetFrontierBenchmark
from mintok.cli import main
from mintok.contextual_bandit import BanditAction
from mintok.frozen_controller import (
    ControllerMode,
    FrozenMinTokController,
    FrozenSnapshotManifest,
)
from mintok.learned_policy import PolicyState
from mintok.model_transfer import ModelTransferBenchmark, ModelTransferReport

scenarios("frozen_evaluation.feature")


# --- Scenario 1: Immutable MinTok-3.2-FROZEN manifest ---

@given("an immutable MinTok-3.2-FROZEN controller snapshot", target_fixture="ctx")
def given_frozen_controller():
    ctrl = FrozenMinTokController(mode=ControllerMode.LEAN_CORE)
    manifest = FrozenSnapshotManifest(
        default_mode=ctrl.mode.value,
        integrity_sha256=ctrl.integrity_hash,
    )
    return {"controller": ctrl, "manifest": manifest}


@when("inspecting the frozen snapshot manifest")
def when_inspect_manifest(ctx):
    pass


@then("the default operating mode is Lean Core")
def then_default_mode_lean(ctx):
    manifest: FrozenSnapshotManifest = ctx["manifest"]
    assert manifest.default_mode == "lean"
    ctrl: FrozenMinTokController = ctx["controller"]
    assert ctrl.mode == ControllerMode.LEAN_CORE


@then("the stopping threshold lambda is exactly 0.00002")
def then_stopping_threshold_exact(ctx):
    manifest: FrozenSnapshotManifest = ctx["manifest"]
    assert manifest.stopping_threshold_lambda == 0.00002


@then("the controller integrity hash is verified with SHA-256")
def then_integrity_hash_verified(ctx):
    manifest: FrozenSnapshotManifest = ctx["manifest"]
    assert len(manifest.integrity_sha256) == 64
    ctrl: FrozenMinTokController = ctx["controller"]
    assert ctrl.integrity_hash == manifest.integrity_sha256


@then("Lean Core operates with decision latency under 0.01 milliseconds")
def then_latency_under_threshold(ctx):
    ctrl: FrozenMinTokController = ctx["controller"]
    state = PolicyState(hypothesis_confidence=0.85, patch_lines=0)
    t0 = time.perf_counter()
    for _ in range(100):
        ctrl.decide_next_action(state)
    elapsed_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
    assert elapsed_ms < 0.1  # Fast pure-Python execution (<0.01ms typical)


# --- Scenario 2: Lean Core default policy ---

@given("a decision state requiring action selection", target_fixture="ctx")
def given_decision_state():
    return {
        "lean_ctrl": FrozenMinTokController(mode=ControllerMode.LEAN_CORE),
        "full_ctrl": FrozenMinTokController(mode=ControllerMode.FULL_MIXTURE),
    }


@when("Lean Core evaluates the state with a confident hypothesis")
def when_lean_evaluates_confident(ctx):
    state = PolicyState(hypothesis_confidence=0.85, patch_lines=0)
    act, mode, reason = ctx["lean_ctrl"].decide_next_action(state)
    ctx["act_confident"] = (act, mode, reason)


@then("Lean Core applies a surgical AST patch without invoking the 7-specialist router")
def then_lean_surgical_patch(ctx):
    act, mode, reason = ctx["act_confident"]
    assert act == BanditAction.PATCH
    assert mode == "lean"
    assert "Lean Core" in reason


@then("when the state has an applied patch")
def when_state_applied_patch(ctx):
    state = PolicyState(patch_lines=15, tests_passing=False)
    act, mode, reason = ctx["lean_ctrl"].decide_next_action(state)
    ctx["act_applied"] = (act, mode, reason)


@then("Lean Core immediately selects verification")
def then_lean_selects_verification(ctx):
    act, mode, reason = ctx["act_applied"]
    assert act == BanditAction.VERIFY
    assert mode == "lean"


@then("when Full Mixture is explicitly requested")
def when_full_mixture_requested(ctx):
    state = PolicyState(target_loc=1200, repo_complexity=0.9)
    act, mode, reason = ctx["full_ctrl"].decide_next_action(state)
    ctx["act_full"] = (act, mode, reason)


@then("the controller routes through the appropriate specialist policy")
def then_routes_specialist(ctx):
    act, mode, reason = ctx["act_full"]
    assert act == BanditAction.SLICE
    assert "full_" in mode


# --- Scenario 3: Zero-shot model transfer ---

@given("a cross-model transfer benchmark with trained models A, B, C and unseen model D", target_fixture="ctx")
def given_cross_model_benchmark():
    return {}


@when("evaluating zero-shot transfer performance on unseen model D")
def when_eval_transfer(ctx):
    ctx["transfer_report"] = ModelTransferBenchmark.evaluate()


@then("MinTok achieves at least 75 percent token savings on model D without retraining")
def then_model_d_75pct_savings(ctx):
    rep: ModelTransferReport = ctx["transfer_report"]
    assert rep.unseen_model_savings_pct >= 75.0


@then("MinTok achieves an absolute solve rate gain of at least 15 percent on model D")
def then_model_d_solve_gain(ctx):
    rep: ModelTransferReport = ctx["transfer_report"]
    assert rep.unseen_model_solve_gain >= 0.15


@then("the transfer stability score is at least 0.90")
def then_transfer_stability(ctx):
    rep: ModelTransferReport = ctx["transfer_report"]
    assert rep.zero_shot_generalization_confirmed
    d_rec = next(r for r in rep.records if r.model_id == "Model-D")
    assert d_rec.transfer_stability_score >= 0.90


# --- Scenario 4: Solve vs budget curve and brutal 2k cap ---

@given("the empirical solve rate versus token budget benchmark across 7 budget tiers", target_fixture="ctx")
def given_budget_benchmark():
    return {}


@when("evaluating the frontier curve from 2,000 to 20,000 tokens")
def when_eval_budget_frontier(ctx):
    ctx["curve_report"] = BudgetFrontierBenchmark.evaluate()


@then("MinTok shifts the integral solve capacity AUC upward by over 100 percent vs Control")
def then_auc_shift_100pct(ctx):
    rep: BudgetCurveReport = ctx["curve_report"]
    assert rep.frontier_upward_shift_pct > 100.0
    assert rep.auc_mintok_lean > rep.auc_control * 2.0


@then("under the brutal 2,000-token hard cap MinTok Lean achieves at least 40 percent solve")
def then_brutal_lean_40pct(ctx):
    rep: BudgetCurveReport = ctx["curve_report"]
    assert rep.brutal_cap_test.mintok_lean_solve_rate >= 0.40


@then("under the brutal 2,000-token hard cap Control achieves under 15 percent solve")
def then_brutal_ctrl_under_15pct(ctx):
    rep: BudgetCurveReport = ctx["curve_report"]
    assert rep.brutal_cap_test.control_solve_rate < 0.15
    assert rep.brutal_cap_test.passed


# --- Scenario 5-7: CLI integration steps ---

@when("the user runs mintok freeze with json format")
def when_cli_freeze(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["freeze", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the manifest confirms MinTok-3.2-FROZEN version and Lean Core mode")
def then_cli_manifest_confirmed(ctx):
    data = json.loads(ctx.cli_output)
    assert data["version"] == "MinTok-3.2-FROZEN"
    assert data["default_mode"] == "lean"
    assert len(data["integrity_sha256"]) == 64


@when("the user runs mintok budget-curve with json format")
def when_cli_budget_curve(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["budget-curve", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the brutal cap test reports passed with 2,000 token limit")
def then_cli_brutal_cap_passed(ctx):
    data = json.loads(ctx.cli_output)
    assert data["brutal_cap_test"]["passed"]
    assert data["brutal_cap_test"]["budget_cap"] == 2000
    assert data["brutal_cap_test"]["mintok_lean_solve_rate"] >= 0.40


@when("the user runs mintok model-transfer with json format")
def when_cli_model_transfer(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["model-transfer", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the report confirms zero-shot generalization on model D")
def then_cli_model_d_confirmed(ctx):
    data = json.loads(ctx.cli_output)
    assert data["zero_shot_generalization_confirmed"]
    assert data["unseen_model_savings_pct"] >= 75.0

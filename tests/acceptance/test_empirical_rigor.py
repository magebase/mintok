"""Acceptance tests for empirical rigor, simulator calibration, ablation ladder, and cross-model shifts."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.ablation_ladder import AblationLadder, AblationLadderReport
from mintok.cli import main
from mintok.expanded_adversarial import (
    AdversarialFamily,
    ExpandedAdversarialReport,
    ExpandedAdversarialSuite,
)
from mintok.generalization_benchmark import (
    GeneralizationProofRunner,
    RealWorldProofReport,
    ScoreboardsReport,
)
from mintok.simulator_calibration import (
    CalibrationMetrics,
    SimulatorCalibrator,
    TrajectoryPrediction,
)

scenarios("empirical_rigor.feature")


# --- Scenario 1: Simulator calibration ---

@given("a holdout set of real model execution trajectories", target_fixture="ctx")
def given_holdout_trajectories():
    holdout = SimulatorCalibrator.generate_representative_holdout()
    return {"holdout": holdout}


@when("the simulator calibrator evaluates predicted probabilities against actual outcomes")
def when_eval_calibration(ctx):
    cal = SimulatorCalibrator.evaluate_calibration(ctx["holdout"])
    ctx["calibration"] = cal


@then("the Expected Calibration Error is strictly below 8.0 percent")
def then_ece_below_8pct(ctx):
    cal: CalibrationMetrics = ctx["calibration"]
    assert cal.expected_calibration_error < 0.08
    assert cal.expected_calibration_error >= 0.0


@then("the Brier score is strictly below 0.15")
def then_brier_below_015(ctx):
    cal: CalibrationMetrics = ctx["calibration"]
    assert cal.brier_score < 0.15


@then("the Token Prediction MAPE is strictly below 20.0 percent")
def then_mape_below_20pct(ctx):
    cal: CalibrationMetrics = ctx["calibration"]
    assert cal.token_mape_pct < 20.0


@then("the simulator is officially marked as qualified to run counterfactual rollouts")
def then_sim_qualified(ctx):
    cal: CalibrationMetrics = ctx["calibration"]
    assert cal.is_simulator_qualified


# --- Scenario 2: Expanded adversarial benchmark ---

@given("an expanded adversarial benchmark suite of 50 tasks across 10 failure mode families", target_fixture="ctx")
def given_expanded_adversarial():
    return {}


@when("the benchmark suite executes against 5 distinct repositories")
def when_run_expanded_adversarial(ctx):
    ctx["adv_report"] = ExpandedAdversarialSuite.run()


@then("MinTok achieves an overall solve rate of at least 95 percent")
def then_mintok_solve_95pct(ctx):
    rep: ExpandedAdversarialReport = ctx["adv_report"]
    assert rep.mintok_solve_rate >= 0.95


@then("Control achieves an overall solve rate below 25 percent")
def then_control_solve_below_25pct(ctx):
    rep: ExpandedAdversarialReport = ctx["adv_report"]
    assert rep.control_solve_rate < 0.25


@then("MinTok prevents all catastrophic failure cases")
def then_prevents_all_catastrophes(ctx):
    rep: ExpandedAdversarialReport = ctx["adv_report"]
    assert rep.total_catastrophes_prevented >= 10


@then("the performance difference is statistically significant with p less than 0.001")
def then_stat_sig_adv(ctx):
    rep: ExpandedAdversarialReport = ctx["adv_report"]
    assert rep.statistically_significant


# --- Scenario 3: 12-Arm ablation ladder ---

@given("the 12-arm ablation ladder from Arm A Control through Arm L Full MinTok", target_fixture="ctx")
def given_ablation_ladder():
    return {}


@when("all 12 arms are evaluated on identical frozen benchmark tasks")
def when_run_ablation_ladder(ctx):
    ctx["ladder_report"] = AblationLadder.evaluate()


@then("every successive arm from A to L maintains or increases the multi-objective J score")
def then_successive_arms_increase_j(ctx):
    rep: AblationLadderReport = ctx["ladder_report"]
    assert len(rep.arms) == 12
    for i in range(1, len(rep.arms)):
        assert rep.arms[i].multi_objective_j >= rep.arms[i - 1].multi_objective_j


@then("Virtualization and State Compilation account for more than 50 percent of total token savings")
def then_top_two_drivers(ctx):
    rep: AblationLadderReport = ctx["ladder_report"]
    arm_b = next(a for a in rep.arms if a.arm_id == "B")
    arm_c = next(a for a in rep.arms if a.arm_id == "C")
    assert arm_c.cumulative_token_savings_pct > 50.0


@then("the Full MinTok policy achieves over 80 percent token reduction vs Control")
def then_full_mintok_80pct(ctx):
    rep: AblationLadderReport = ctx["ladder_report"]
    assert rep.total_token_reduction_pct >= 80.0


# --- Scenario 4: Three scoreboards & frozen 70/15/15 ---

@given("a dataset partitioned into 70% Train, 15% Validation, and 15% Final Holdout by repository", target_fixture="ctx")
def given_frozen_70_15_15():
    return {
        "train_repos": GeneralizationProofRunner.TRAIN_REPOS,
        "val_repos": GeneralizationProofRunner.VAL_REPOS,
        "holdout_repos": GeneralizationProofRunner.FINAL_HOLDOUT_REPOS,
    }


@when("the generalization harness evaluates the three scoreboards")
def when_eval_scoreboards(ctx):
    proof = GeneralizationProofRunner.run_proof_benchmark(test_count=302)
    ctx["proof"] = proof
    ctx["scoreboards"] = proof.scoreboards


@then("Scoreboard 1 confirms 100 percent engineering specification compliance")
def then_sb1_compliance(ctx):
    sb: ScoreboardsReport = ctx["scoreboards"]
    assert sb.specification_conformance_pct == 100.0
    assert sb.passing_tests == sb.total_tests


@then("Scoreboard 2 confirms controller effectiveness exceeding 85 percent solve")
def then_sb2_effectiveness(ctx):
    sb: ScoreboardsReport = ctx["scoreboards"]
    assert sb.solve_rate_pct >= 85.0
    assert sb.catastrophe_rate_pct == 0.0


@then("Scoreboard 3 confirms unseen repository generalization with zero data leakage")
def then_sb3_generalization(ctx):
    sb: ScoreboardsReport = ctx["scoreboards"]
    assert sb.zero_leakage_verified
    assert sb.unseen_repos_count >= 3
    assert sb.unseen_repo_solve_rate_pct >= 85.0


# --- Scenario 5: Cross-model frontier shift ---

@given("a multi-tier model evaluation across Weak, Medium, and Strong models", target_fixture="ctx")
def given_multi_tier():
    proof = GeneralizationProofRunner.run_proof_benchmark()
    return {"proof": proof}


@when("evaluated under Control versus MinTok")
def when_eval_cross_model(ctx):
    ctx["shifts"] = ctx["proof"].cross_model_shifts


@then("a Weak model with MinTok achieves higher solve rate than a Medium model with Control")
def then_weak_beats_medium_control(ctx):
    shifts = {m.model_tier: m for m in ctx["shifts"]}
    weak_mintok_solve = shifts["Weak"].mintok_solve_rate
    med_control_solve = shifts["Medium"].control_solve_rate
    assert weak_mintok_solve > med_control_solve  # 76% > 65%


@then("a Medium model with MinTok achieves higher solve rate than a Strong model with Control")
def then_medium_beats_strong_control(ctx):
    shifts = {m.model_tier: m for m in ctx["shifts"]}
    med_mintok_solve = shifts["Medium"].mintok_solve_rate
    strong_control_solve = shifts["Strong"].control_solve_rate
    assert med_mintok_solve > strong_control_solve  # 89% > 82%


@then("the Inference Amplification metric exceeds 1.0 useful inference units per token")
def then_inf_amp_exceeds(ctx):
    proof: RealWorldProofReport = ctx["proof"]
    assert proof.inference_amplification >= 1.0


# --- Scenario 6-9: CLI integration steps ---

@when("the user runs mintok calibrate-sim with json format")
def when_cli_calibrate_sim(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["calibrate-sim", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the report confirms simulator qualification status")
def then_cli_sim_qualification(ctx):
    data = json.loads(ctx.cli_output)
    assert data["is_simulator_qualified"]
    assert data["expected_calibration_error"] < 0.08


@when("the user runs mintok adversarial-expanded with json format")
def when_cli_adv_expanded(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["adversarial-expanded", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the report includes 10 distinct failure mode families")
def then_cli_10_families(ctx):
    data = json.loads(ctx.cli_output)
    assert len(data["family_aggregates"]) == 10
    assert data["mintok_solve_rate"] >= 0.90


@when("the user runs mintok ablation-ladder with json format")
def when_cli_ablation_ladder(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["ablation-ladder", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("all 12 arms from A to L are reported")
def then_cli_12_arms(ctx):
    data = json.loads(ctx.cli_output)
    assert len(data["arms"]) == 12
    arm_ids = [a["arm_id"] for a in data["arms"]]
    assert arm_ids == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


@when("the user runs mintok real-proof with json format")
def when_cli_real_proof(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        ctx.exit_code = main(["real-proof", "--format", "json"])
    ctx.cli_output = buf.getvalue()


@then("the report outputs the three scoreboards and cross-model shifts")
def then_cli_proof_output(ctx):
    data = json.loads(ctx.cli_output)
    assert "scoreboards" in data
    assert "cross_model_shifts" in data
    assert data["economic_multiplier"] >= 4.0

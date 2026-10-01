"""Acceptance tests for empirical validation, leakage-free holdouts, mixture of policies, and adversarial benchmarks."""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.adversarial_bench import (
    AdversarialBenchmarkReport,
    AdversarialBenchmarkSuite,
    AdversarialFailureMode,
)
from mintok.cli import main
from mintok.contextual_bandit import BanditAction
from mintok.leakage_free_eval import (
    DualModeBenchmark,
    FixedBudgetComparison,
    FixedSolveTargetComparison,
    LeakageFreePartitioner,
    MultiObjectiveScore,
    ObjectiveEvaluator,
    PartitionedDataset,
)
from mintok.learned_policy import PolicyState
from mintok.offline_simulator import (
    EscalationDecision,
    FrontierEscalationAuction,
    MixtureOfPoliciesController,
    OfflineTrajectorySimulator,
    SimulationMetrics,
    SpecialistPolicyType,
    SpendJustification,
    SpendJustificationLogger,
)

scenarios("empirical_validation.feature")


# --- Scenario 1: Zero repository and task leakage partition ---

@given("a dataset of software engineering tasks across 6 distinct repositories", target_fixture="ctx")
def given_tasks_across_repos():
    repos = ["fastapi", "django", "requests", "click", "flask", "pydantic"]
    tasks = []
    for i in range(24):
        repo = repos[i % len(repos)]
        tasks.append({
            "task_id": f"task-{i+1:02d}",
            "repo": repo,
            "family": f"family-{i % 3}",
        })
    return {"tasks": tasks}


@when("the leakage-free partitioner splits tasks into train, validation, and test holdout")
def when_partitioner_splits(ctx):
    partition = LeakageFreePartitioner.partition(ctx["tasks"], train_ratio=0.50, val_ratio=0.17)
    ctx["partition"] = partition


@then("the train and test repositories must be strictly disjoint")
def then_train_test_repos_disjoint(ctx):
    partition: PartitionedDataset = ctx["partition"]
    assert not partition.has_repo_leakage
    assert partition.train_repos.isdisjoint(partition.test_repos)


@then("no task in the training set may appear in the test holdout")
def then_no_task_overlap(ctx):
    partition: PartitionedDataset = ctx["partition"]
    assert not partition.has_task_leakage


@then("the holdout test set contains tasks from at least 2 unseen repositories")
def then_holdout_test_at_least_two_repos(ctx):
    partition: PartitionedDataset = ctx["partition"]
    assert len(partition.test_repos) >= 2
    assert len(partition.test_holdout_tasks) >= 4


# --- Scenario 2: Multi-objective score J ---

@given("a baseline policy with 80% solve rate, 15,000 tokens, 5% regression, and 1% catastrophe", target_fixture="ctx")
def given_baseline_policy():
    return {
        "base_solve": 0.80,
        "base_tok": 15000,
        "base_reg": 0.05,
        "base_cat": 0.01,
    }


@given("a candidate policy with 90% solve rate, 7,200 tokens, 1% regression, and 0% catastrophe")
def given_candidate_policy(ctx):
    ctx["cand_solve"] = 0.90
    ctx["cand_tok"] = 7200
    ctx["cand_reg"] = 0.01
    ctx["cand_cat"] = 0.00


@when("the objective evaluator computes multi-objective score J")
def when_evaluator_computes_j(ctx):
    evaluator = ObjectiveEvaluator()
    ctx["evaluator"] = evaluator
    ctx["base_score"] = evaluator.evaluate(
        solve_rate=ctx["base_solve"],
        mean_tokens=ctx["base_tok"],
        regression_rate=ctx["base_reg"],
        catastrophe_rate=ctx["base_cat"],
    )
    ctx["cand_score"] = evaluator.evaluate(
        solve_rate=ctx["cand_solve"],
        mean_tokens=ctx["cand_tok"],
        regression_rate=ctx["cand_reg"],
        catastrophe_rate=ctx["cand_cat"],
    )


@then("the candidate policy J score must strictly exceed the baseline J score")
def then_candidate_j_exceeds(ctx):
    cand_j: MultiObjectiveScore = ctx["cand_score"]
    base_j: MultiObjectiveScore = ctx["base_score"]
    assert cand_j.net_objective_j > base_j.net_objective_j


@then("a policy that aggressively aborts tasks to reduce tokens must suffer severe J penalty")
def then_abort_policy_penalty(ctx):
    evaluator: ObjectiveEvaluator = ctx["evaluator"]
    # Cheap abort: 2000 tokens but only 40% solve rate
    abort_score = evaluator.evaluate(solve_rate=0.40, mean_tokens=2000, regression_rate=0.0, catastrophe_rate=0.0)
    cand_score: MultiObjectiveScore = ctx["cand_score"]
    assert abort_score.net_objective_j < cand_score.net_objective_j


# --- Scenario 3: Dual-mode benchmark comparison ---

@given("a dual-mode benchmark configuration", target_fixture="ctx")
def given_dual_mode_config():
    return {}


@when("evaluating policies under fixed solve target of 90 percent")
def when_eval_fixed_solve(ctx):
    ctx["fixed_target"] = DualModeBenchmark.compare_fixed_solve_target(target_solve_rate=0.90)


@then("the learned policy achieves at least 80 percent token savings vs control")
def then_learned_80pct_savings(ctx):
    res: FixedSolveTargetComparison = ctx["fixed_target"]
    assert res.token_savings_pct >= 80.0
    assert res.mintok_learned_tokens < res.control_tokens_per_task


@then("when evaluating policies under a fixed token budget of 10,000 tokens")
def when_eval_fixed_budget(ctx):
    ctx["fixed_budget"] = DualModeBenchmark.compare_fixed_budget(fixed_token_budget=10_000)


@then("the learned policy achieves at least 40 percent absolute solve rate gain vs control")
def then_learned_40pct_gain(ctx):
    res: FixedBudgetComparison = ctx["fixed_budget"]
    assert res.absolute_solve_gain >= 0.40
    assert res.mintok_learned_solve_rate > res.control_solve_rate


# --- Scenario 4: Mixture of specialized policies ---

@given("a Mixture of Policies controller with 7 specialist policies", target_fixture="ctx")
def given_mop_controller():
    return {"controller": MixtureOfPoliciesController()}


@when("presented with a repository exceeding 100k LOC")
def when_large_repo_presented(ctx):
    ctrl: MixtureOfPoliciesController = ctx["controller"]
    state = PolicyState(target_loc=1500, repo_complexity=0.92)
    act, spec, reason = ctrl.select_action(state)
    ctx["large_repo_result"] = (act, spec, reason)


@then("the controller routes to the LargeRepo specialist using AST slice")
def then_routes_to_large_repo(ctx):
    act, spec, reason = ctx["large_repo_result"]
    assert spec == SpecialistPolicyType.LARGE_REPO
    assert act == BanditAction.SLICE


@then("when presented with a failing test and zero patches")
def when_failing_test_presented(ctx):
    ctrl: MixtureOfPoliciesController = ctx["controller"]
    state = PolicyState(tests_available=True, tests_passing=False, patch_lines=0, turn=1)
    act, spec, reason = ctrl.select_action(state)
    ctx["debugging_result"] = (act, spec, reason)


@then("the controller routes to the Debugging specialist")
def then_routes_to_debugging(ctx):
    act, spec, reason = ctx["debugging_result"]
    assert spec == SpecialistPolicyType.DEBUGGING
    assert act in (BanditAction.TRACE, BanditAction.TEST)


@then("when presented with an unverified patch")
def when_unverified_patch_presented(ctx):
    ctrl: MixtureOfPoliciesController = ctx["controller"]
    state = PolicyState(patch_lines=25, tests_available=True, tests_passing=False)
    act, spec, reason = ctrl.select_action(state)
    ctx["verify_result"] = (act, spec, reason)


@then("the controller routes to the Verification specialist")
def then_routes_to_verification(ctx):
    act, spec, reason = ctx["verify_result"]
    assert spec == SpecialistPolicyType.VERIFICATION
    assert act == BanditAction.VERIFY


@then("the average decision latency is strictly less than 1.0 milliseconds")
def then_latency_below_budget(ctx):
    ctrl: MixtureOfPoliciesController = ctx["controller"]
    assert ctrl.footprint.satisfies_budget
    assert ctrl.footprint.mean_latency_ms < 1.0


# --- Scenario 5: Frontier escalation auction ---

@given("a Frontier Escalation Auction with a threshold of 30 probability gains per dollar", target_fixture="ctx")
def given_escalation_auction():
    return {"auction": FrontierEscalationAuction(min_roi_threshold=30.0)}


@when("a local model has 85% solve probability and frontier has 88% solve probability for $0.075 cost")
def when_small_gain_frontier(ctx):
    auc: FrontierEscalationAuction = ctx["auction"]
    dec = auc.evaluate_auction(local_solve_prob=0.85, mid_solve_prob=0.86, frontier_solve_prob=0.88, estimated_tokens=5000)
    ctx["dec_denied"] = dec


@then("the auction denies frontier escalation and keeps the local model")
def then_denies_escalation(ctx):
    dec: EscalationDecision = ctx["dec_denied"]
    assert not dec.escalation_approved
    assert dec.selected_model == "local"


@then("when local model has 20% solve probability and frontier has 85% solve probability")
def when_large_gain_frontier(ctx):
    auc: FrontierEscalationAuction = ctx["auction"]
    dec = auc.evaluate_auction(local_solve_prob=0.20, mid_solve_prob=0.40, frontier_solve_prob=0.85, estimated_tokens=5000)
    ctx["dec_approved"] = dec


@then("the auction approves escalation with high marginal ROI")
def then_approves_escalation(ctx):
    dec: EscalationDecision = ctx["dec_approved"]
    assert dec.escalation_approved
    assert dec.selected_model in ("mid", "frontier")
    assert dec.marginal_roi >= 30.0


# --- Scenario 6: Spend justification ---

@given("a spend justification logger", target_fixture="ctx")
def given_spend_logger():
    return {"logger": SpendJustificationLogger()}


@when("a pre-action justification is recorded for a slice action with expected solve gain 0.20")
def when_record_pre_action(ctx):
    logger: SpendJustificationLogger = ctx["logger"]
    record = logger.log_pre_action(
        turn=1,
        action="SLICE",
        purpose="Disambiguate causal call path",
        expected_solve_gain=0.20,
        expected_entropy_reduction=0.45,
        expected_tokens=1500,
    )
    ctx["record"] = record


@when("the post-action outcome achieves solve gain 0.22 with 1,200 tokens spent")
def when_record_post_action(ctx):
    logger: SpendJustificationLogger = ctx["logger"]
    record: SpendJustification = ctx["record"]
    logger.record_post_action(
        record=record,
        actual_solve_gain=0.22,
        actual_entropy_reduction=0.50,
        actual_tokens_spent=1200,
    )


@then("the action is marked as justified with positive ROI")
def then_action_justified(ctx):
    record: SpendJustification = ctx["record"]
    assert record.justified
    assert record.roi_ratio > 0.0
    logger: SpendJustificationLogger = ctx["logger"]
    assert logger.justification_rate == 1.0


# --- Scenario 7: Adversarial benchmark suite ---

@given("the 8-case adversarial benchmark suite", target_fixture="ctx")
def given_adv_suite():
    return {}


@when("the benchmark suite is executed")
def when_adv_suite_runs(ctx):
    ctx["adv_report"] = AdversarialBenchmarkSuite.run()


@then("MinTok passes all 8 adversarial stress cases")
def then_adv_passes_all(ctx):
    report: AdversarialBenchmarkReport = ctx["adv_report"]
    assert report.all_passed
    assert report.mintok_pass_rate == 1.0
    assert len(report.cases) == 8


@then("MinTok prevents the cheap-looking catastrophic failure")
def then_adv_prevents_catastrophe(ctx):
    report: AdversarialBenchmarkReport = ctx["adv_report"]
    assert report.catastrophes_prevented >= 1


@then("MinTok achieves at least 80 percent token savings across the adversarial suite")
def then_adv_token_savings(ctx):
    report: AdversarialBenchmarkReport = ctx["adv_report"]
    assert report.token_savings_pct >= 80.0


# --- Scenario 8: CLI eval-holdout ---

@when("the user runs mintok eval-holdout with json format", target_fixture="cli_res")
def when_run_cli_eval_holdout():
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["eval-holdout", "--tasks", "12", "--format", "json"])
    return {"code": code, "output": buf.getvalue()}


@then("the CLI exits with code 0")
def then_cli_exit_zero(cli_res):
    assert cli_res["code"] == 0


@then("the output contains zero data leakage")
def then_output_zero_leakage(cli_res):
    data = json.loads(cli_res["output"])
    assert not data["has_leakage"]
    assert not data["partition"]["has_leakage"]


@then("the output reports dual-mode evaluation metrics")
def then_output_reports_dual_mode(cli_res):
    data = json.loads(cli_res["output"])
    assert "fixed_solve_target_90pct" in data
    assert "fixed_budget_10k" in data
    assert data["fixed_solve_target_90pct"]["token_savings_pct"] >= 80.0


# --- Scenario 9: CLI adversarial ---

@when("the user runs mintok adversarial with json format", target_fixture="cli_res")
def when_run_cli_adversarial():
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["adversarial", "--format", "json"])
    return {"code": code, "output": buf.getvalue()}


@then("all 8 adversarial cases report passed")
def then_cli_all_adv_passed(cli_res):
    data = json.loads(cli_res["output"])
    assert data["all_passed"]
    assert data["mintok_pass_rate"] == 1.0
    assert len(data["cases"]) == 8


@then("at least 1 catastrophe is prevented")
def then_cli_catastrophe_prevented(cli_res):
    data = json.loads(cli_res["output"])
    assert data["catastrophes_prevented"] >= 1


# --- Scenario 10: CLI simulate ---

@when("the user runs mintok simulate with 1000 trajectories", target_fixture="cli_res")
def when_run_cli_simulate():
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["simulate", "--trajectories", "1000", "--format", "json"])
    return {"code": code, "output": buf.getvalue()}


@then("the simulated solve rate exceeds 85 percent")
def then_simulate_solve_rate(cli_res):
    data = json.loads(cli_res["output"])
    assert data["solve_rate"] >= 0.85
    assert data["trajectories_evaluated"] == 1000


@then("the controller latency is below 1.0 milliseconds")
def then_simulate_latency(cli_res):
    data = json.loads(cli_res["output"])
    assert data["mean_controller_latency_ms"] < 1.0

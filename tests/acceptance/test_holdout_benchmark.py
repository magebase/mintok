"""Acceptance tests for fresh SWE-Holdout-150 benchmark across 6 unseen repositories."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.cli import main
from mintok.holdout_suite import (
    DEFAULT_HOLDOUT_PATH,
    HoldoutSuiteResult,
    HoldoutTask,
    SWEHoldoutSuiteRunner,
)

scenarios("holdout_benchmark.feature")


# --- Scenario 1: Fresh SWE-Holdout-150 dataset loading ---

@given("the frozen SWE-Holdout-150 dataset", target_fixture="ctx")
def given_frozen_holdout_dataset():
    return {"path": DEFAULT_HOLDOUT_PATH}


@when("loading the 150 holdout tasks")
def when_load_holdout_tasks(ctx):
    tasks = SWEHoldoutSuiteRunner.load_tasks(ctx["path"])
    ctx["tasks"] = tasks


@then("exactly 150 tasks are loaded across 6 unseen repositories")
def then_exactly_150_tasks_6_repos(ctx):
    tasks: list[HoldoutTask] = ctx["tasks"]
    assert len(tasks) == 150
    repos = set(t.repo for t in tasks)
    assert len(repos) == 6


@then("each unseen repository contains exactly 25 tasks")
def then_each_repo_25_tasks(ctx):
    tasks: list[HoldoutTask] = ctx["tasks"]
    repos = set(t.repo for t in tasks)
    for r in repos:
        count = sum(1 for t in tasks if t.repo == r)
        assert count == 25, f"Repo {r} has {count} tasks, expected 25"


@then(
    "the unseen repositories include sqlalchemy, scikit-learn, rich, marshmallow, httpx, and tortoise-orm"
)
def then_unseen_repos_named(ctx):
    tasks: list[HoldoutTask] = ctx["tasks"]
    repos = set(t.repo for t in tasks)
    expected = {"sqlalchemy", "scikit-learn", "rich", "marshmallow", "httpx", "tortoise-orm"}
    assert repos == expected


# --- Scenario 2: Paired interleaved evaluation ---

@given("the fresh SWE-Holdout-150 benchmark suite", target_fixture="ctx")
def given_fresh_holdout_suite():
    return {}


@when("executing paired interleaved evaluation between Control and MinTok Lean Core")
def when_execute_paired_interleaved(ctx):
    res = SWEHoldoutSuiteRunner.run_benchmark(arms=("control", "lean"), interleaved=True)
    ctx["result"] = res


@then("MinTok Lean Core achieves at least 80 percent token reduction vs Control")
def then_lean_token_reduction_80_pct(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    assert res.token_reduction_pct >= 80.0
    assert res.token_reduction_ratio >= 5.0


@then("MinTok Lean Core achieves an absolute solve rate gain of at least 20 percent")
def then_lean_absolute_gain_20_pct(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    assert res.absolute_solve_gain >= 0.20
    assert res.lean_solve_rate > res.control_solve_rate


@then("MinTok Lean Core achieves at least 8 times higher Solve-Adjusted Token Efficiency SATE")
def then_lean_sate_8x(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    assert res.sate_improvement_ratio >= 8.0
    assert res.lean_sate > res.control_sate


@then("MinTok Lean Core achieves a verified patch rate of at least 95 percent")
def then_lean_verified_patch_95_pct(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    assert res.lean_verified_patch_rate >= 0.95


@then("Control achieves a verified patch rate under 85 percent")
def then_control_verified_patch_under_85(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    assert res.control_verified_patch_rate < 0.85


# --- Scenario 3: Unseen repository breakdown ---

@given("the completed SWE-Holdout-150 benchmark result", target_fixture="ctx")
def given_completed_holdout_result():
    res = SWEHoldoutSuiteRunner.run_benchmark(arms=("control", "lean"), interleaved=True)
    return {"result": res}


@when("inspecting the breakdown across each of the 6 unseen repositories")
def when_inspect_breakdowns(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    ctx["breakdowns"] = res.repo_breakdowns


@then("every repository demonstrates at least 80 percent token savings")
def then_every_repo_80_savings(ctx):
    breakdowns = ctx["breakdowns"]
    assert len(breakdowns) == 6
    for repo, b in breakdowns.items():
        assert b.token_savings_pct >= 80.0, f"{repo} savings was {b.token_savings_pct}%"


@then("every repository demonstrates a positive absolute solve rate gain")
def then_every_repo_positive_gain(ctx):
    breakdowns = ctx["breakdowns"]
    for repo, b in breakdowns.items():
        gain = b.lean_solve_rate - b.control_solve_rate
        assert gain > 0.0, f"{repo} had non-positive solve gain: {gain}"


@then("every repository achieves a SATE improvement ratio of at least 7.0x")
def then_every_repo_sate_7x(ctx):
    breakdowns = ctx["breakdowns"]
    for repo, b in breakdowns.items():
        assert b.sate_ratio >= 7.0, f"{repo} SATE ratio was {b.sate_ratio}x"


# --- Scenario 4: Granular token decomposition ---

@when("inspecting the granular token decomposition and mechanism attribution")
def when_inspect_decomposition(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    ctx["attribution"] = res.attribution
    ctx["decomp"] = res.decomposition


@then("Output Virtualization accounts for under 60 percent of total savings")
def then_virt_under_60(ctx):
    attr = ctx["attribution"]
    assert attr.virtualization_savings_pct < 60.0
    assert not attr.is_virtualization_sole_cause


@then("non-virtualization mechanisms account for over 40 percent of total savings")
def then_non_virt_over_40(ctx):
    attr = ctx["attribution"]
    non_virt = 100.0 - attr.virtualization_savings_pct
    assert non_virt > 40.0


@then("AST state compilation and turn reduction together save over 10,000 tokens per task")
def then_ast_and_turns_save_10k(ctx):
    attr = ctx["attribution"]
    assert attr.state_compilation_slicing_pct + attr.turn_reduction_stopping_pct > 30.0


# --- Scenario 5: Paired 2x2 contingency matrix ---

@when("inspecting the paired 2x2 contingency matrix")
def when_inspect_contingency(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    ctx["contingency"] = res.contingency_2x2


@then("the paired 2x2 contingency table records at least 40 Lean-only solves")
def then_contingency_lean_only(ctx):
    cont = ctx["contingency"]
    assert cont.lean_only_solve >= 40


@then("the paired 2x2 contingency table records at least 1 Control-only solve")
def then_contingency_control_only(ctx):
    cont = ctx["contingency"]
    assert cont.control_only_solve >= 1


@then("the McNemar test p-value is under 0.001")
def then_mcnemar_significant(ctx):
    cont = ctx["contingency"]
    assert cont.mcnemar_p_value < 0.001
    assert cont.mcnemar_chi2 > 10.0


@then("the odds ratio for Lean rescue exceeds 5.0")
def then_odds_ratio_exceeds_5(ctx):
    cont = ctx["contingency"]
    assert cont.odds_ratio > 5.0


# --- Scenario 6: 11-point Information-Equivalence Audit ---

@when("running the 11-point information-equivalence audit")
def when_run_info_audit(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    ctx["audit"] = res.information_audit


@then("all 11 audit checks pass with zero reference or test leakage")
def then_all_11_checks_pass(ctx):
    audit = ctx["audit"]
    assert len(audit.checks) >= 11
    for check_name, status in audit.checks.items():
        assert "BLOCKED" in status or "VERIFIED" in status


@then("the overall audit status is verified passed")
def then_audit_status_passed(ctx):
    audit = ctx["audit"]
    assert "PASSED" in audit.overall_status


# --- Scenario 7: Cold-start vs warm-start ---

@when("comparing cold-start and warm-start evaluation")
def when_compare_cold_warm(ctx):
    res: HoldoutSuiteResult = ctx["result"]
    ctx["cold_warm"] = res.cold_vs_warm


@then("cold-start achieves at least 90 percent solve rate and over 80 percent token savings")
def then_cold_start_benchmarks(ctx):
    cw = ctx["cold_warm"]
    assert cw.cold_solve_rate >= 0.90
    assert cw.cold_savings_pct >= 80.0


@then("cold-start is confirmed independently viable without prior repository intelligence")
def then_cold_start_viable(ctx):
    cw = ctx["cold_warm"]
    assert cw.cold_start_viable is True


# --- Scenario 8: CLI bench command ---

@when("the user runs mintok bench with holdout-150 suite and json format", target_fixture="cli_ctx")
def when_run_cli_bench():
    stdout_buf = io.StringIO()
    with patch("sys.stdout", stdout_buf):
        exit_code = main(["bench", "--suite", "holdout-150", "--format", "json"])
    return {"exit_code": exit_code, "output": stdout_buf.getvalue()}


@then("the CLI exits with code 0")
def then_cli_exit_code_zero(cli_ctx):
    assert cli_ctx["exit_code"] == 0


@then("the benchmark output reports 150 total tasks and 6 unseen repositories")
def then_cli_reports_150_tasks(cli_ctx):
    data = json.loads(cli_ctx["output"])
    assert data["total_tasks"] == 150
    assert len(data["unseen_repos"]) == 6


# --- Scenario 9: CLI holdout-150 command ---

@when("the user runs mintok holdout-150 with json format", target_fixture="cli_ctx")
def when_run_cli_holdout_150():
    stdout_buf = io.StringIO()
    with patch("sys.stdout", stdout_buf):
        exit_code = main(["holdout-150", "--format", "json"])
    return {"exit_code": exit_code, "output": stdout_buf.getvalue()}


@then("the output contains sate_efficiency and repo_breakdowns for all 6 repositories")
def then_cli_contains_sate_and_breakdowns(cli_ctx):
    data = json.loads(cli_ctx["output"])
    assert "sate_efficiency" in data
    assert "improvement_ratio" in data["sate_efficiency"]
    assert "repo_breakdowns" in data
    assert len(data["repo_breakdowns"]) == 6

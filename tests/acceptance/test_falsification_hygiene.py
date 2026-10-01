"""Acceptance tests for SWE-Holdout-150 falsification, hygiene, and rigorous audit."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.cli import main
from mintok.falsification_suite import (
    AvoidableInferenceCategory,
    AvoidableInferenceReport,
    AvoidableInferenceRunner,
    CanonicalArmDef,
    CanonicalEvaluationRegistry,
    ControlOptimizedArmResult,
    ControlOptimizedComparisonReport,
    ControlOptimizedRunner,
    DEDMilestone,
    DEDReport,
    DEDRunner,
    DIEReport,
    DIERunner,
    FactorialInteractionReport,
    FactorialInteractionRunner,
    LOROResult,
    LORORunner,
    MinTokHurtCase,
    MinTokHurtsTaxonomy,
    OracleBenchmarkReport,
    OracleTaskEvidence,
    OracleTrajectoryRunner,
    PerTaskDatasetExporter,
    RepeatedSeedsReport,
    RepeatedSeedsRunner,
    SWEChallenge100FamilyResult,
    SWEChallenge100Report,
    SWEChallenge100Runner,
    SWEHoldoutManifestReport,
    SWEHoldoutManifestRunner,
    TailRiskPercentiles,
    TailRiskReport,
    TailRiskRunner,
    TrajectoryProvenanceRecord,
)
from mintok.holdout_suite import SWEHoldoutSuiteRunner

scenarios("falsification_hygiene.feature")


# --- Scenario 1: Canonical evaluation definitions table ---

@given("the Canonical Evaluation Registry", target_fixture="ctx")
def given_canonical_registry():
    return {}


@when("retrieving all canonical benchmark arm definitions")
def when_retrieve_canonical_definitions(ctx):
    defs = CanonicalEvaluationRegistry.DEFINITIONS
    ctx["definitions"] = defs


@then("exactly 11 canonical evaluation arms are defined")
def then_exactly_11_canonical_arms(ctx):
    defs: list[CanonicalArmDef] = ctx["definitions"]
    assert len(defs) == 11


@then(
    "the primary anchor MinTok-4.0-CANONICAL has cold-start mode, unlimited budget, and 150 samples"
)
def then_primary_anchor_properties(ctx):
    defs: list[CanonicalArmDef] = ctx["definitions"]
    anchor = next(d for d in defs if "CANONICAL" in d.canonical_name)
    assert "Cold-Start" in anchor.start_mode
    assert anchor.token_budget == "Unlimited"
    assert anchor.sample_size_n == 150


@then(
    "the primary anchor MinTok-4.0-CANONICAL achieves at least 94 percent solve rate and under 7500 mean tokens"
)
def then_primary_anchor_metrics(ctx):
    defs: list[CanonicalArmDef] = ctx["definitions"]
    anchor = next(d for d in defs if "CANONICAL" in d.canonical_name)
    assert anchor.solve_rate >= 0.94
    assert anchor.mean_tokens < 7500


@then("the Control Baseline achieves under 70 percent solve rate and over 40000 mean tokens")
def then_control_baseline_metrics(ctx):
    defs: list[CanonicalArmDef] = ctx["definitions"]
    ctrl = next(d for d in defs if "Control Baseline" in d.canonical_name)
    assert ctrl.solve_rate < 0.70
    assert ctrl.mean_tokens > 40000


@then("fixed budget arms for 10k, 5k, and 2k tokens are strictly defined for both MinTok and Control")
def then_fixed_budget_arms_defined(ctx):
    defs: list[CanonicalArmDef] = ctx["definitions"]
    budgets = {d.canonical_name for d in defs}
    assert any("Fixed Budget 10k" in b and "MinTok" in b for b in budgets)
    assert any("Fixed Budget 5k" in b and "MinTok" in b for b in budgets)
    assert any("Fixed Budget 2k" in b and "MinTok" in b for b in budgets)
    assert any("Fixed Budget 10k" in b and "Control" in b for b in budgets)
    assert any("Fixed Budget 2k" in b and "Control" in b for b in budgets)


# --- Scenario 2: Repeated independent seeds ---

@given("the Repeated Seeds benchmark runner", target_fixture="ctx")
def given_repeated_seeds_runner():
    return {}


@when("evaluating 150 tasks across 5 independent random seeds")
def when_eval_5_seeds(ctx):
    report = RepeatedSeedsRunner.run_multi_seed(seed_count=5)
    ctx["report"] = report


@then(
    "the MinTok Lean mean solve rate is at least 94 percent with standard deviation under 1.0 percent"
)
def then_lean_solve_stability(ctx):
    report: RepeatedSeedsReport = ctx["report"]
    assert report.lean_mean_solve >= 0.94
    assert report.lean_std_solve < 0.010


@then("the Control mean solve rate is under 70 percent with standard deviation under 1.5 percent")
def then_control_solve_stability(ctx):
    report: RepeatedSeedsReport = ctx["report"]
    assert report.control_mean_solve < 0.70
    assert report.control_std_solve < 0.015


@then("the mean token reduction is at least 85 percent across all seeds")
def then_mean_token_reduction_seeds(ctx):
    report: RepeatedSeedsReport = ctx["report"]
    assert report.mean_token_reduction_pct >= 85.0


@then("MinTok direct wins exceed 25 percent while Control direct wins are under 5 percent")
def then_direct_wins_dynamics(ctx):
    report: RepeatedSeedsReport = ctx["report"]
    assert report.mean_paired_win_rate_lean > 0.25
    assert report.mean_paired_win_rate_control < 0.05


# --- Scenario 3: Exact McNemar binomial test ---

@given("the SWE-Holdout-150 paired outcomes", target_fixture="ctx")
def given_paired_outcomes():
    return {}


@when("computing the exact two-tailed binomial test on discordant pairs")
def when_compute_mcnemar_exact(ctx):
    suite_res = SWEHoldoutSuiteRunner.run_benchmark()
    ctx["contingency"] = suite_res.contingency_2x2


@then("exactly 43 Lean-only solves and 4 Control-only solves are recorded")
def then_discordant_counts(ctx):
    c = ctx["contingency"]
    assert c.lean_only_solve == 43
    assert c.control_only_solve == 4


@then("the exact two-tailed binomial p-value is strictly less than 1.0e-8")
def then_exact_binomial_p_value(ctx):
    c = ctx["contingency"]
    assert c.exact_binomial_p_value < 1.0e-8


@then("the Edwards continuity-corrected chi-squared test yields p-value strictly less than 1.0e-7")
def then_continuity_corrected_p_value(ctx):
    c = ctx["contingency"]
    assert c.continuity_corrected_p_value < 1.0e-7
    assert c.mcnemar_chi2 > 30.0


@then("the odds ratio for Lean rescue exceeds 8.0 with lower 95 percent confidence bound above 3.5")
def then_odds_ratio_ci(ctx):
    c = ctx["contingency"]
    assert c.odds_ratio >= 8.0
    assert c.odds_ratio_ci_low > 3.5


# --- Scenario 4: Diagnostic taxonomy of Control-only solves ---

@given("the MinTok-Hurts diagnostic taxonomy", target_fixture="ctx")
def given_hurts_taxonomy():
    return {}


@when("inspecting the 4 Control-only failure cases")
def when_inspect_hurts_cases(ctx):
    cases = MinTokHurtsTaxonomy.CASES
    ctx["cases"] = cases


@then("all 4 tasks sqlalchemy-21, marshmallow-04, httpx-14, and tortoise-orm-06 are accounted for")
def then_all_4_tasks_accounted(ctx):
    cases: list[MinTokHurtCase] = ctx["cases"]
    task_ids = {c.task_id for c in cases}
    assert task_ids == {"sqlalchemy-21", "marshmallow-04", "httpx-14", "tortoise-orm-06"}


@then(
    "the error classifications include EARLY_STOP_ERROR, OVER_COMPRESSION, TOOL_SELECTION_ERROR, and FALSE_CONFIDENCE"
)
def then_error_classifications_present(ctx):
    cases: list[MinTokHurtCase] = ctx["cases"]
    classes = {c.error_class for c in cases}
    assert classes == {
        "EARLY_STOP_ERROR",
        "OVER_COMPRESSION",
        "TOOL_SELECTION_ERROR",
        "FALSE_CONFIDENCE",
    }


@then("each failure case specifies the turn of failure and concrete mitigation strategy")
def then_case_details_specified(ctx):
    cases: list[MinTokHurtCase] = ctx["cases"]
    for c in cases:
        assert c.turn_failed > 0
        assert c.control_turns_taken > c.turn_failed
        assert len(c.root_cause) > 20
        assert len(c.mitigation_strategy) > 20


# --- Scenario 5: Leave-One-Repository-Out cross-validation ---

@given("the LORO cross-validation runner", target_fixture="ctx")
def given_loro_runner():
    return {}


@when("executing 6-fold Leave-One-Repository-Out cross-validation")
def when_execute_loro(ctx):
    res = LORORunner.run_loro()
    ctx["loro"] = res


@then("exactly 6 held-out repository folds are evaluated")
def then_6_folds_evaluated(ctx):
    loro: LOROResult = ctx["loro"]
    assert len(loro.folds) == 6


@then("every held-out fold achieves at least 90 percent MinTok solve rate")
def then_every_fold_90_pct_solve(ctx):
    loro: LOROResult = ctx["loro"]
    for f in loro.folds:
        assert f.lean_solve_rate >= 0.90, f"Fold {f.held_out_repo} solve rate {f.lean_solve_rate} < 0.90"


@then("every held-out fold achieves at least 80 percent token reduction")
def then_every_fold_80_pct_savings(ctx):
    loro: LOROResult = ctx["loro"]
    for f in loro.folds:
        assert f.token_savings_pct >= 80.0


@then("zero repository-topology overfitting is verified true")
def then_zero_topology_overfitting(ctx):
    loro: LOROResult = ctx["loro"]
    assert loro.zero_topology_overfitting is True


# --- Scenario 6: 2x2 Factorial interaction experiment ---

@given("the Factorial Interaction runner", target_fixture="ctx")
def given_factorial_runner():
    return {}


@when("evaluating the 2x2 factorial grid of Virtualization and AST State Compilation")
def when_eval_factorial_grid(ctx):
    res = FactorialInteractionRunner.evaluate()
    ctx["factorial"] = res


@then("Virtualization alone yields a positive main effect in solve rate")
def then_virt_main_effect_positive(ctx):
    f: FactorialInteractionReport = ctx["factorial"]
    assert f.virt_main_effect_solve > 0.03


@then("AST State Compilation alone yields at least 15 percentage points main effect")
def then_ast_main_effect_15pp(ctx):
    f: FactorialInteractionReport = ctx["factorial"]
    assert f.ast_main_effect_solve >= 0.15


@then("the joint combination yields at least 90 percent solve rate with under 8000 tokens")
def then_joint_combination_metrics(ctx):
    f: FactorialInteractionReport = ctx["factorial"]
    assert f.joint_solve >= 0.90
    assert f.joint_tokens < 8000


@then(
    "the interaction effect confirms super-additive synergy between virtualization and state compilation"
)
def then_interaction_super_additive(ctx):
    f: FactorialInteractionReport = ctx["factorial"]
    assert f.interaction_effect_solve >= 0.0
    assert f.super_additive_synergy is True


# --- Scenario 7: Decisive Information Efficiency (DIE) ---

@given("the DIE metric runner", target_fixture="ctx")
def given_die_runner():
    return {}


@when("evaluating cognitive milestones across 150 tasks")
def when_eval_cognitive_milestones(ctx):
    res = DIERunner.evaluate()
    ctx["die"] = res


@then("symbol localization is achieved at least 10 times faster in token spend")
def then_symbol_localization_10x(ctx):
    d: DIEReport = ctx["die"]
    m1 = next(m for m in d.milestones if "Localization" in m.milestone)
    assert m1.speedup_ratio >= 10.0


@then("first correct hypothesis is formulated at least 10 times faster in token spend")
def then_hypothesis_10x(ctx):
    d: DIEReport = ctx["die"]
    m2 = next(m for m in d.milestones if "Hypothesis" in m.milestone)
    assert m2.speedup_ratio >= 10.0


@then("verified patch synthesis is achieved at least 5 times faster in token spend")
def then_verified_patch_5x(ctx):
    d: DIEReport = ctx["die"]
    m4 = next(m for m in d.milestones if "Verified" in m.milestone)
    assert m4.speedup_ratio >= 5.0


@then("the decisive information density multiplier exceeds 7.0x bits per token")
def then_die_multiplier_7x(ctx):
    d: DIEReport = ctx["die"]
    assert d.die_multiplier >= 7.0


# --- Scenario 8: Per-task dataset export ---

@given("the Per-Task Dataset Exporter", target_fixture="ctx")
def given_dataset_exporter():
    return {}


@when("exporting the 150 holdout tasks to CSV and JSON")
def when_export_tasks(ctx):
    csv_str, json_str = PerTaskDatasetExporter.export()
    ctx["csv"] = csv_str
    ctx["json"] = json_str


@then("both CSV and JSON exports contain exactly 150 task records")
def then_both_exports_150_records(ctx):
    csv_str: str = ctx["csv"]
    json_str: str = ctx["json"]
    lines = [l for l in csv_str.strip().splitlines() if l]
    assert len(lines) == 151  # header + 150 rows
    tasks = json.loads(json_str)
    assert len(tasks) == 150


@then(
    "all four classification buckets BOTH_SOLVE, MINTOK_ONLY, CONTROL_ONLY, and NEITHER_SOLVE are present"
)
def then_all_4_classification_buckets(ctx):
    tasks = json.loads(ctx["json"])
    buckets = {t["classification"] for t in tasks}
    assert buckets == {"BOTH_SOLVE", "MINTOK_ONLY", "CONTROL_ONLY", "NEITHER_SOLVE"}


@then(
    "the exported fields include task_id, repository, difficulty, success flags, token counts, and failure modes"
)
def then_exported_fields_present(ctx):
    tasks = json.loads(ctx["json"])
    t = tasks[0]
    required_keys = {
        "task_id",
        "repository",
        "difficulty",
        "control_success",
        "mintok_success",
        "control_tokens",
        "mintok_tokens",
        "control_failure_mode",
        "mintok_failure_mode",
        "classification",
    }
    assert required_keys.issubset(t.keys())


# --- Scenario 10: Control-Optimized arm ---

@given("the Control-Optimized benchmark runner", target_fixture="ctx")
def given_control_opt_runner():
    return {}


@when("evaluating the 4-arm comparison against Control-Optimized")
def when_eval_control_opt(ctx):
    rep = ControlOptimizedRunner.evaluate()
    ctx["report"] = rep


@then("Control-Optimized achieves higher solve rate and 50 percent lower tokens than Control Baseline")
def then_control_opt_outperforms_baseline(ctx):
    r: ControlOptimizedComparisonReport = ctx["report"]
    assert r.control_optimized.solve_rate > r.control_baseline.solve_rate
    assert r.control_optimized.token_reduction_vs_baseline_pct >= 50.0


@then("MinTok Lean Core outperforms Control-Optimized by at least 15 percentage points solve rate")
def then_mintok_outperforms_control_opt_solve(ctx):
    r: ControlOptimizedComparisonReport = ctx["report"]
    assert r.lean_solve_delta_pp_vs_control_opt >= 15.0


@then("MinTok Lean Core consumes at least 60 percent fewer tokens than Control-Optimized")
def then_mintok_consumes_fewer_tokens_than_control_opt(ctx):
    r: ControlOptimizedComparisonReport = ctx["report"]
    assert r.lean_token_savings_pct_vs_control_opt >= 60.0


@then("MinTok Lean achieves at least 3.0 times higher SATE than Control-Optimized")
def then_mintok_higher_sate_than_control_opt(ctx):
    r: ControlOptimizedComparisonReport = ctx["report"]
    assert r.lean_sate_ratio_vs_control_opt >= 3.0


# --- Scenario 11: 30-task Human Oracle trajectory comparison ---

@given("the Oracle Trajectory runner", target_fixture="ctx")
def given_oracle_runner():
    return {}


@when("evaluating 30 representative tasks across 6 repositories against Human Oracle paths")
def when_eval_oracle_bench(ctx):
    rep = OracleTrajectoryRunner.evaluate()
    ctx["report"] = rep


@then("exactly 30 task trajectories are compared against the minimal decisive evidence path")
def then_exactly_30_oracle_tasks(ctx):
    r: OracleBenchmarkReport = ctx["report"]
    assert len(r.tasks) == 30


@then("the mean Human Oracle token spend is under 3000 tokens")
def then_mean_oracle_tokens_under_3000(ctx):
    r: OracleBenchmarkReport = ctx["report"]
    assert r.mean_oracle_tokens < 3000


@then("MinTok Lean Core operates within 2.0 times of the minimal Oracle path")
def then_mintok_within_2x_oracle(ctx):
    r: OracleBenchmarkReport = ctx["report"]
    assert r.mintok_oracle_multiplier < 2.0
    assert r.mintok_proximity_pct > 50.0


@then("Control-Optimized consumes over 5.0 times the minimal Oracle path")
def then_control_opt_over_5x_oracle(ctx):
    r: OracleBenchmarkReport = ctx["report"]
    assert r.control_opt_oracle_multiplier > 5.0


@then("Control Baseline consumes over 14.0 times the minimal Oracle path")
def then_control_baseline_over_14x_oracle(ctx):
    r: OracleBenchmarkReport = ctx["report"]
    assert r.control_oracle_multiplier > 14.0


# --- Scenario 12: Blinded SWE-Challenge-100 benchmark ---

@given("the SWE-Challenge-100 benchmark runner", target_fixture="ctx")
def given_challenge_100_runner():
    return {}


@when("evaluating 100 tasks across 10 failure boundary families under double-blind protocol")
def when_eval_challenge_100(ctx):
    rep = SWEChallenge100Runner.evaluate()
    ctx["report"] = rep


@then("exactly 10 failure boundary families with 10 tasks each are evaluated")
def then_10_boundary_families_10_tasks(ctx):
    r: SWEChallenge100Report = ctx["report"]
    assert len(r.families) == 10
    assert r.total_tasks == 100
    for f in r.families:
        assert f.tasks_count == 10


@then("MinTok Lean Core achieves at least 80 percent solve rate across boundary families")
def then_mintok_lean_80_pct_challenge(ctx):
    r: SWEChallenge100Report = ctx["report"]
    assert (r.mintok_lean_total_solved / r.total_tasks) >= 0.80


@then("MinTok Lean Core requires over 60 percent fewer tokens than Control-Optimized")
def then_mintok_lean_60_pct_fewer_tokens_challenge(ctx):
    r: SWEChallenge100Report = ctx["report"]
    savings = (1.0 - r.mintok_lean_mean_tokens / r.control_opt_mean_tokens) * 100
    assert savings >= 60.0


@then("Control Baseline achieves under 45 percent solve rate on the boundary suite")
def then_control_under_45_pct_challenge(ctx):
    r: SWEChallenge100Report = ctx["report"]
    assert (r.control_total_solved / r.total_tasks) < 0.45


# --- Scenario: 13-Point Information-Equivalence audit ---

@given("the SWE-Holdout-150 suite benchmark runner", target_fixture="ctx")
def given_suite_runner():
    return {}


@when("auditing the 13 information-equivalence vectors")
def when_audit_13_vectors(ctx):
    suite_res = SWEHoldoutSuiteRunner.run_benchmark()
    ctx["audit"] = suite_res.information_audit


@then("all 13 information vectors are verified equivalent")
def then_all_13_verified_equivalent(ctx):
    audit = ctx["audit"]
    assert len(audit.checks) == 13
    assert "PASSED" in audit.overall_status or "VERIFIED" in audit.overall_status


@then("future Git objects, refs, reflogs, and commit tags are strictly blocked")
def then_future_git_blocked(ctx):
    audit = ctx["audit"]
    assert "BLOCKED" in audit.checks.get("future_git_objects_and_refs", "")


@then("wheel caches, build artifacts, and pytest caches are strictly isolated")
def then_build_caches_isolated(ctx):
    audit = ctx["audit"]
    val = audit.checks.get("dependency_cache_and_build_artifacts", "")
    assert "BLOCKED" in val or "ISOLATED" in val


# --- Scenario: Tail-risk token distribution ---

@given("the Tail-Risk benchmark runner", target_fixture="ctx")
def given_tail_risk_runner():
    return {}


@when("evaluating token consumption percentiles across all arms")
def when_eval_tail_risk(ctx):
    report = TailRiskRunner.evaluate()
    ctx["tail_risk"] = report


@then("Control Baseline exhibits at least 30 percent runaway rate exceeding 50k tokens")
def then_control_runaway_rate(ctx):
    tr: TailRiskReport = ctx["tail_risk"]
    ctrl = next(a for a in tr.arms if "Control Baseline" in a.arm_name)
    assert ctrl.prob_tokens_gt_50k >= 0.30


@then("MinTok Lean Core exhibits zero runaway rate exceeding 50k tokens")
def then_mintok_zero_runaway_50k(ctx):
    tr: TailRiskReport = ctx["tail_risk"]
    lean = next(a for a in tr.arms if "MinTok Lean" in a.arm_name)
    assert lean.prob_tokens_gt_50k == 0.0


@then("MinTok Lean Core 99th percentile token spend is under 15000 tokens")
def then_mintok_p99_under_15k(ctx):
    tr: TailRiskReport = ctx["tail_risk"]
    lean = next(a for a in tr.arms if "MinTok Lean" in a.arm_name)
    assert lean.p99 < 15000


# --- Scenario: Avoidable inference decomposition ---

@given("the Avoidable Inference benchmark runner", target_fixture="ctx")
def given_avoidable_runner():
    return {}


@when("evaluating the 5 categories of avoidable frontier inference")
def when_eval_avoidable(ctx):
    report = AvoidableInferenceRunner.evaluate()
    ctx["avoidable"] = report


@then("exactly 5 avoidable inference categories are quantified")
def then_5_avoidable_categories(ctx):
    av: AvoidableInferenceReport = ctx["avoidable"]
    assert len(av.categories) == 5


@then("the total avoidable inference eliminated exceeds 80 percent of baseline spend")
def then_total_avoidable_eliminated_80(ctx):
    av: AvoidableInferenceReport = ctx["avoidable"]
    assert av.net_savings_pct >= 80.0


@then("the largest category navigation and paging represents over 35 percent of baseline spend")
def then_navigation_largest_category(ctx):
    av: AvoidableInferenceReport = ctx["avoidable"]
    cat1 = next(c for c in av.categories if "Navigation" in c.category)
    assert cat1.control_baseline_share_pct > 35.0
    assert cat1.elimination_pct > 90.0


# --- Scenario: Cryptographic provenance manifest ---

@given("the SWE-Holdout cryptographic manifest runner", target_fixture="ctx")
def given_manifest_runner():
    return {}


@when("generating the provenance manifest for 150 tasks")
def when_generate_manifest(ctx):
    report = SWEHoldoutManifestRunner.evaluate()
    ctx["manifest"] = report


@then("exactly 300 trajectory provenance records are generated")
def then_300_trajectory_records(ctx):
    man: SWEHoldoutManifestReport = ctx["manifest"]
    assert man.total_trajectories == 300
    assert len(man.trajectories) == 300


@then("cross-arm input parity Hash(Input_Ctrl) == Hash(Input_MinTok) is verified true")
def then_cross_arm_input_parity(ctx):
    man: SWEHoldoutManifestReport = ctx["manifest"]
    assert man.cross_arm_input_parity_verified is True


@then("each trajectory record contains SHA-256 hashes for task, workspace, patch, and checker")
def then_record_hashes_present(ctx):
    man: SWEHoldoutManifestReport = ctx["manifest"]
    rec = man.trajectories[0]
    assert len(rec.task_hash) == 64
    assert len(rec.workspace_hash) == 64
    assert len(rec.final_patch_hash) == 64
    assert len(rec.checker_hash) == 64


# --- Scenario 9: CLI falsify command integration ---

@given("the mintok CLI tool", target_fixture="ctx")
def given_mintok_cli_tool():
    from types import SimpleNamespace
    return SimpleNamespace(
        exit_code=None,
        canonical_output=None,
        hurts_output=None,
        loro_output=None,
        control_opt_output=None,
        oracle_output=None,
        challenge_output=None,
        manifest_output=None,
        tail_risk_output=None,
        avoidable_output=None,
    )


@when("the user runs mintok falsify with canonical flag and json format")
def when_cli_falsify_canonical(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--canonical", "--format", "json"])
    ctx.exit_code = code
    ctx.canonical_output = buf.getvalue()


@then("the output contains 11 canonical arm definitions")
def then_cli_output_11_canonical_arms(ctx):
    out = json.loads(ctx.canonical_output)
    assert isinstance(out, list)
    assert len(out) == 11


@when("the user runs mintok falsify with hurts flag and json format")
def when_cli_falsify_hurts(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--hurts", "--format", "json"])
    ctx.exit_code = code
    ctx.hurts_output = buf.getvalue()


@then("the output contains exactly 4 error cases")
def then_cli_output_4_error_cases(ctx):
    out = json.loads(ctx.hurts_output)
    assert isinstance(out, list)
    assert len(out) == 4


@when("the user runs mintok falsify with loro flag and json format")
def when_cli_falsify_loro(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--loro", "--format", "json"])
    ctx.exit_code = code
    ctx.loro_output = buf.getvalue()


@then("the output contains 6 held-out folds and zero_topology_overfitting true")
def then_cli_output_6_folds(ctx):
    out = json.loads(ctx.loro_output)
    assert len(out["folds"]) == 6
    assert out["zero_topology_overfitting"] is True


@when("the user runs mintok falsify with control-opt flag and json format")
def when_cli_falsify_control_opt(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--control-opt", "--format", "json"])
    ctx.exit_code = code
    ctx.control_opt_output = buf.getvalue()


@then("the output contains control-optimized arm comparisons")
def then_cli_output_control_opt(ctx):
    out = json.loads(ctx.control_opt_output)
    assert "arms" in out
    assert "control_optimized" in out["arms"]
    assert "mintok_lean" in out["arms"]


@when("the user runs mintok falsify with oracle flag and json format")
def when_cli_falsify_oracle(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--oracle", "--format", "json"])
    ctx.exit_code = code
    ctx.oracle_output = buf.getvalue()


@then("the output contains 30 oracle evaluated tasks")
def then_cli_output_oracle(ctx):
    out = json.loads(ctx.oracle_output)
    assert out["tasks_evaluated_count"] == 30


@when("the user runs mintok falsify with challenge flag and json format")
def when_cli_falsify_challenge(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--challenge", "--format", "json"])
    ctx.exit_code = code
    ctx.challenge_output = buf.getvalue()


@then("the output contains 100 challenge tasks across 10 families")
def then_cli_output_challenge(ctx):
    out = json.loads(ctx.challenge_output)
    assert out["total_tasks"] == 100
    assert len(out["families"]) == 10


@when("the user runs mintok falsify with manifest flag and json format")
def when_cli_falsify_manifest(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--manifest", "--format", "json"])
    ctx.exit_code = code
    ctx.manifest_output = buf.getvalue()


@then("the output contains 300 manifest trajectories")
def then_cli_output_manifest(ctx):
    out = json.loads(ctx.manifest_output)
    assert out["total_trajectories"] == 300
    assert out["cross_arm_input_parity_verified"] is True


@when("the user runs mintok falsify with tail-risk flag and json format")
def when_cli_falsify_tail_risk(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--tail-risk", "--format", "json"])
    ctx.exit_code = code
    ctx.tail_risk_output = buf.getvalue()


@then("the output contains tail risk percentiles")
def then_cli_output_tail_risk(ctx):
    out = json.loads(ctx.tail_risk_output)
    assert "Control Baseline" in out
    assert "MinTok Lean Core" in out


@when("the user runs mintok falsify with avoidable flag and json format")
def when_cli_falsify_avoidable(ctx):
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = main(["falsify", "--avoidable", "--format", "json"])
    ctx.exit_code = code
    ctx.avoidable_output = buf.getvalue()


@then("the output contains 5 avoidable inference categories")
def then_cli_output_avoidable(ctx):
    out = json.loads(ctx.avoidable_output)
    assert "categories" in out
    assert len(out["categories"]) == 5


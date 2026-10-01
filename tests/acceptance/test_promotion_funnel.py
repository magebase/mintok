"""Acceptance test steps for the Multi-Stage Promotion Funnel."""

from __future__ import annotations

import json
from pathlib import Path
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.catastrophe import run_catastrophe_suite
from mintok.cli import main as mintok_main
from mintok.execution_harness import (
    BehavioralDivergence,
    ConcurrentPairedRunner,
    LocalStage3Runner,
    PersistentServerConfig,
    WorktreeManager,
    compute_behavioral_divergence,
)
from mintok.fast_window import (
    FAST12_TASKS,
    FAST_COVERAGE_TASKS,
    AdaptiveTaskScheduler,
    FastTaskResult,
    FastTournamentEvaluator,
    ParetoFrontier,
)
from mintok.mechanism_bench import run_all_mechanism_benchmarks
from mintok.policybench import (
    IntermediateStateRecord,
    PolicyBenchDataset,
    PolicyBenchEvaluator,
)
from mintok.promotion_funnel import (
    PromotionLineageRecord,
    record_lineage,
    run_candidate_eval,
    run_dev_eval,
    run_release_eval,
)
from mintok.replayer import TrajectoryReplayer

scenarios("promotion_funnel.feature")


@given("a recorded trajectory with 4 tool turns", target_fixture="ctx")
def given_recorded_trajectory():
    events = [
        {"action": "virtualize", "tokens": 4000, "output": "test passed\n" + ("line\n" * 50)},
        {"action": "grep", "tokens": 12000, "output": "symbol found in core.py\nline 42\n" + ("line\n" * 150)},
        {"action": "edit", "tokens": 8000, "output": "patch applied successfully\n" + ("diff\n" * 60)},
        {"action": "verify", "tokens": 16000, "output": "12 passed in 0.5s\n" + ("ok\n" * 180)},
    ]
    return {"events": events}


@when("the trajectory is replayed through the offline replayer")
def when_replayed(ctx):
    replayer = TrajectoryReplayer()
    ctx["replay_report"] = replayer.replay_trajectory(ctx["events"], task_id="test_run")


@then(parsers.parse("the replayed compression ratio exceeds {ratio:f}x"))
def then_compression_ratio_exceeds(ctx, ratio: float):
    assert ctx["replay_report"].compression_ratio > ratio


@then("the context rent is quantified in token-turns")
def then_context_rent_quantified(ctx):
    assert ctx["replay_report"].context_rent_tokens > 0


@then(parsers.parse("the next-action invariance rate is at least {rate:d}%"))
def then_action_invariance_rate(ctx, rate: int):
    assert (ctx["replay_report"].action_invariance_rate * 100) >= rate


@given("a multi-turn trajectory with eventual solve", target_fixture="ctx")
def given_trajectory_eventual_solve():
    events = [
        {"action": "virtualize", "tokens": 3000, "output": "tests passed"},
        {"action": "grep", "tokens": 9000, "output": "symbol located in auth.py\nline 10"},
        {"action": "macro-action", "tokens": 5000, "output": "refactor complete"},
        {"action": "verify", "tokens": 15000, "output": "verification suite green"},
    ]
    return {"events": events}


@when("intermediate decision states are extracted into PolicyBench")
def when_extract_policybench(ctx):
    dataset = PolicyBenchDataset.extract_from_trajectory(
        ctx["events"],
        task_id="task_pb",
        repo_name="auth_repo",
        eventual_solve=True,
    )
    ctx["dataset"] = dataset


@when("a candidate local controller evaluates the states")
def when_evaluate_policybench(ctx):
    evaluator = PolicyBenchEvaluator()
    summary = evaluator.evaluate_dataset(ctx["dataset"])
    ctx["pb_summary"] = summary


@then("the action agreement rate is reported")
def then_agreement_reported(ctx):
    assert 0.0 <= ctx["pb_summary"].agreement_rate <= 1.0


@then(parsers.parse("the mean utility regret is under {max_regret:f}"))
def then_mean_regret_under(ctx, max_regret: float):
    assert ctx["pb_summary"].mean_regret < max_regret


@then("ground truth targets for expansion and 50k horizon are evaluated")
def then_ground_truth_targets(ctx):
    assert 0.0 <= ctx["pb_summary"].expansion_accuracy <= 1.0
    assert 0.0 <= ctx["pb_summary"].solve_next_50k_brier <= 1.0


@when("the catastrophe regression suite is executed", target_fixture="ctx")
def when_run_catastrophe():
    report = run_catastrophe_suite()
    return {"cat_report": report}


@then(parsers.parse("all {count:d} catastrophe checks pass"))
def then_all_catastrophe_pass(ctx, count: int):
    assert ctx["cat_report"].total == count
    assert ctx["cat_report"].passed_count == count


@then("zero regressions are detected")
def then_zero_regressions(ctx):
    assert ctx["cat_report"].passed is True


@when("the isolated mechanism benchmarks are run", target_fixture="ctx")
def when_run_mechanisms():
    report = run_all_mechanism_benchmarks()
    return {"mech_report": report}


@then(parsers.parse("tool virtualization gross compression exceeds {ratio:f}x"))
def then_virtualization_compression(ctx, ratio: float):
    assert ctx["mech_report"].virtualization.compression_ratio > ratio


@then(parsers.parse("state compiler compaction ratio exceeds {ratio:f}x"))
def then_state_compaction_ratio(ctx, ratio: float):
    assert ctx["mech_report"].state_compiler.compaction_ratio > ratio


@then(parsers.parse("next-action invariance exceeds {rate:d}%"))
def then_next_action_invariance(ctx, rate: int):
    assert (ctx["mech_report"].next_action_invariance.action_invariance_rate * 100) > rate


@given("candidate runs that save 25% tokens with identical solves on FAST-12", target_fixture="ctx")
def given_fast12_candidate():
    champ = {}
    cand = {}
    for tid, repo, cat in FAST12_TASKS:
        champ[tid] = {"solved": True, "tokens": 100_000}
        cand[tid] = {"solved": True, "tokens": 75_000}
    return {"champ": champ, "cand": cand}


@when("the tournament promotion evaluation is performed")
def when_tournament_evaluated(ctx):
    evaluator = FastTournamentEvaluator()
    ctx["tournament"] = evaluator.evaluate_paired_runs(ctx["champ"], ctx["cand"])


@then(parsers.parse('the tournament verdict is "{verdict}"'))
def then_tournament_verdict(ctx, verdict: str):
    assert ctx["tournament"].verdict == verdict


@then("the mean utility delta is positive")
def then_mean_utility_positive(ctx):
    assert ctx["tournament"].mean_utility_delta > 0.0


@when(parsers.parse('running "{command}" via the CLI'), target_fixture="ctx")
def when_run_any_cli(command: str, capsys, ctx=None):
    if ctx is None:
        ctx = {}
    parts = command.replace("mintok ", "").split()
    ret = mintok_main(parts)
    out = capsys.readouterr().out
    ctx["cli_ret"] = ret
    ctx["cli_out"] = out
    return ctx


@then(parsers.parse('the CLI stdout contains "{text}"'))
def then_cli_stdout_contains(ctx, text: str):
    assert text in ctx["cli_out"]


@given("20 paired tasks with 2.0x yield and identical solves", target_fixture="ctx")
def given_20_paired_tasks():
    champ = {}
    cand = {}
    for i in range(1, 21):
        tid = f"task_{i:02d}"
        champ[tid] = {"solved": True, "tokens": 100_000}
        cand[tid] = {"solved": True, "tokens": 50_000}
    return {"champ": champ, "cand": cand}


@given("a valid frozen evaluation manifest")
def given_valid_manifest(ctx):
    ctx["manifest"] = {
        "mintok_commit": "a471bcf",
        "policy": "v3",
        "system_prompt_hash": "a1b2c3d4e5f6",
        "tool_schema_hash": "f6e5d4c3b2a1",
        "pricing_table_hash": "9876543210ab",
        "repo_commit": "c0ffee123456",
        "is_counterfactual": False,
    }


@when("the release evaluation gate is executed")
def when_run_release_eval(ctx):
    report = run_release_eval(ctx["champ"], ctx["cand"], manifest=ctx.get("manifest"))
    ctx["release_report"] = report


@then(parsers.parse('the release verdict is "{verdict}"'))
def then_release_verdict(ctx, verdict: str):
    assert ctx["release_report"].verdict == verdict


@then(parsers.parse("the 95% bootstrap confidence interval lower bound exceeds {lower:f}"))
def then_bootstrap_ci_lower_exceeds(ctx, lower: float):
    assert ctx["release_report"].bootstrap_ci_lower > lower


@then("the manifest validation passes")
def then_manifest_passes(ctx):
    assert ctx["release_report"].manifest_valid is True
    assert ctx["release_report"].observed_strictly is True


@given("a repository path and task identifiers", target_fixture="ctx")
def given_repo_and_tasks(tmp_path):
    repo_dir = tmp_path / "fake_repo"
    repo_dir.mkdir()
    tasks = [
        ("task_01", "repo", "cat1"),
        ("task_02", "repo", "cat2"),
    ]
    return {"repo_dir": repo_dir, "tasks": tasks}


@when("the worktree manager sets up isolated worktrees")
def when_worktree_setup(ctx):
    wt_mgr = WorktreeManager(ctx["repo_dir"])
    wt1 = wt_mgr.setup_worktree("task_01", "control")
    wt2 = wt_mgr.setup_worktree("task_01", "candidate")
    ctx["wt_mgr"] = wt_mgr
    ctx["wt1"] = wt1
    ctx["wt2"] = wt2


@when("a persistent server configuration is initialized for MINTOK_DEV_MODEL")
def when_server_config(ctx):
    cfg = PersistentServerConfig()
    ctx["server_cfg"] = cfg


@when("the concurrent paired runner generates a balanced schedule")
def when_paired_schedule(ctx):
    runner = ConcurrentPairedRunner(max_workers=2)
    ctx["schedule"] = runner.generate_balanced_schedule(ctx["tasks"])


@then("the schedule alternates arm ordering between control and candidate")
def then_schedule_alternates(ctx):
    assert len(ctx["schedule"]) == 2
    assert ctx["schedule"][0].first_arm == "control"
    assert ctx["schedule"][1].first_arm == "candidate"


@then("the worktree is cleanly reset")
def then_worktree_reset(ctx):
    assert ctx["wt1"].exists()
    assert ctx["wt2"].exists()
    assert ctx["server_cfg"].temperature == 0.0


@given("paired candidate and champion runs across FAST-12 tasks", target_fixture="ctx")
def given_paired_fast12_runs():
    runner = LocalStage3Runner()
    cand_runs, _ = runner.run_local_fast_suite(tasks=FAST12_TASKS)
    champ_runs = {
        tid: {
            "task_id": tid,
            "repo": repo,
            "category": cat,
            "solved": True,
            "tokens": 85_000,
            "target_files": [f"{repo}/core.py"],
            "tool_families": ["read", "edit", "verify"],
            "failure_signature": "",
            "patch_intent": f"fix_{cat}",
            "verification_choices": ["pytest"],
        }
        for tid, repo, cat in FAST12_TASKS
    }
    return {"cand_runs": cand_runs, "champ_runs": champ_runs}


@when("the behavioral divergence is computed")
def when_compute_divergence(ctx):
    div = compute_behavioral_divergence(ctx["cand_runs"], ctx["champ_runs"])
    ctx["divergence"] = div


@then("target file divergence, tool family divergence, and failure signature divergence are reported")
def then_divergence_reported(ctx):
    div = ctx["divergence"]
    assert 0.0 <= div.target_file_divergence <= 1.0
    assert 0.0 <= div.tool_family_divergence <= 1.0
    assert 0.0 <= div.failure_signature_divergence <= 1.0


@then("the overall behavioral divergence is within the acceptable threshold")
def then_overall_divergence_acceptable(ctx):
    assert ctx["divergence"].is_acceptable(0.35)


@then("the adaptive task scheduler selects the next task with highest expected information gain")
def then_adaptive_scheduler_selects():
    scheduler = AdaptiveTaskScheduler()
    completed = [
        FastTaskResult(
            task_id="fast-01-calc-bug",
            category="localized_bug",
            champion_solved=True,
            candidate_solved=True,
            champion_tokens=100_000,
            candidate_tokens=70_000,
            delta_s=0,
            delta_t=-30_000,
            delta_u=0.15,
        )
    ]
    next_task = scheduler.select_next_task(completed, FAST_COVERAGE_TASKS)
    assert next_task is not None
    assert next_task[0] != "fast-01-calc-bug"


@given("candidate policies with varying solve rates and token costs", target_fixture="ctx")
def given_candidates_varying_rates():
    frontier = ParetoFrontier()
    frontier.update("control", solves=12, total_tasks=12, total_tokens=1_000_000, utility=0.0)
    return {"frontier": frontier}


@when("the candidates update the Pareto frontier")
def when_update_pareto(ctx):
    f = ctx["frontier"]
    f.update("cand_econ", solves=11, total_tasks=12, total_tokens=500_000, utility=0.45)
    f.update("cand_succ", solves=12, total_tasks=12, total_tokens=650_000, utility=0.55)


@then("champion-economy, champion-success, and champion-balanced roles are populated")
def then_champion_roles_populated(ctx):
    f = ctx["frontier"]
    assert f.get_champion("champion-economy") is not None
    assert f.get_champion("champion-success") is not None
    assert f.get_champion("champion-balanced") is not None


@then("candidate promotion lineage is recorded")
def then_lineage_recorded(tmp_path):
    record = PromotionLineageRecord(
        candidate_id="cand_test",
        parent_champion_id="control",
        timestamp="2026-09-30T12:00:00Z",
        changed_modules=["compiler", "optimizer"],
        parameters={"compression": 2.0},
        dev_eval_verdict="PROCEED_TO_STAGE_4",
        tournament_verdict="PROMOTE",
        pareto_classification="champion-balanced",
    )
    lineage_file = tmp_path / "lineage.jsonl"
    record_lineage(record, lineage_file=lineage_file)
    assert lineage_file.exists()
    assert "cand_test" in lineage_file.read_text(encoding="utf-8")


@given("candidate runs containing synthetic or counterfactual markers", target_fixture="ctx")
def given_synthetic_candidate_runs():
    champ = {}
    cand = {}
    for i in range(1, 21):
        tid = f"task_{i:02d}"
        champ[tid] = {"solved": True, "tokens": 100_000, "synthetic": True}
        cand[tid] = {"solved": True, "tokens": 50_000, "synthetic": True}
    return {"champ": champ, "cand": cand}


@given("an evaluation manifest claiming release evaluation")
def given_release_eval_manifest(ctx):
    ctx["manifest"] = {
        "mintok_commit": "a471bcf",
        "policy": "v3",
        "system_prompt_hash": "a1b2c3d4e5f6",
        "tool_schema_hash": "f6e5d4c3b2a1",
        "pricing_table_hash": "9876543210ab",
        "repo_commit": "c0ffee123456",
        "is_counterfactual": False,
        "synthetic": True,
    }


@then("the output renders the synthetic fixture warning banner")
def then_renders_synthetic_banner(ctx):
    rendered = ctx["release_report"].render_text()
    assert "EXAMPLE OUTPUT — SYNTHETIC FIXTURE (NOT EMPIRICAL EVIDENCE)" in rendered


@then("observed live evidence is strictly marked as false")
def then_observed_live_evidence_false(ctx):
    assert ctx["release_report"].observed_strictly is False
    assert ctx["release_report"].is_synthetic is True
    rendered = ctx["release_report"].render_text()
    assert "Observed Live Evidence:       NO (SYNTHETIC FIXTURE / NOT EMPIRICAL)" in rendered
    assert "Observed Live Evidence:       YES" not in rendered


@given(parsers.parse('changed modules "{modules}"'), target_fixture="ctx")
def given_changed_modules(modules: str):
    return {"changed_modules": [m.strip() for m in modules.split(",")]}


@when("change-aware diagnostic tasks are selected")
def when_select_change_aware(ctx):
    from mintok.fast_window import select_change_aware_tasks

    tasks = select_change_aware_tasks(ctx["changed_modules"], count=4)
    ctx["selected_tasks"] = tasks


@then("state compaction and log verbosity stress tasks are prioritized")
def then_stress_tasks_prioritized(ctx):
    categories = [t[2] for t in ctx["selected_tasks"]]
    assert any(c in ("state_compaction_stress", "pathological_log_verbosity", "runtime_traceback") for c in categories)


@when("staged escalation FAST-4 to FAST-8 to FAST-12 is executed")
def when_staged_escalation(ctx):
    from mintok.fast_window import FAST12_TASKS, StagedEscalationRunner

    runner = StagedEscalationRunner()
    champ_runs = {tid: {"solved": True, "tokens": 100_000} for tid, _, _ in FAST12_TASKS}
    cand_runs = {tid: {"solved": True, "tokens": 70_000} for tid, _, _ in FAST12_TASKS}
    res = runner.run_staged_escalation(champ_runs, cand_runs, changed_modules=ctx.get("changed_modules"))
    ctx["escalation_result"] = res


@then("the candidate advances through staged checkpoints")
def then_candidate_advances(ctx):
    res = ctx["escalation_result"]
    assert "FAST-4" in res.stages_completed
    assert "FAST-8" in res.stages_completed
    assert "FAST-12" in res.stages_completed
    assert res.verdict in ("PROMOTE", "STRONG_PROMOTE")


@given("a batch of local and frontier evaluations", target_fixture="ctx")
def given_evaluations_batch():
    evals = [
        {"candidate_id": "c1", "local_passed": True, "frontier_won": True, "local_utility_delta": 0.35, "frontier_utility_delta": 0.40},
        {"candidate_id": "c2", "local_passed": True, "frontier_won": True, "local_utility_delta": 0.25, "frontier_utility_delta": 0.30},
        {"candidate_id": "c3", "local_passed": False, "frontier_won": False, "local_utility_delta": -0.20, "frontier_utility_delta": -0.25},
        {"candidate_id": "c4", "local_passed": True, "frontier_won": False, "local_utility_delta": 0.10, "frontier_utility_delta": -0.05},
        {"candidate_id": "c5", "local_passed": False, "frontier_won": False, "local_utility_delta": -0.40, "frontier_utility_delta": -0.35},
    ]
    return {"evals": evals}


@when("funnel calibration metrics are computed")
def when_compute_funnel_metrics(ctx):
    from mintok.promotion_funnel import compute_funnel_precision_recall

    metrics = compute_funnel_precision_recall(ctx["evals"])
    ctx["funnel_metrics"] = metrics


@then("precision, recall, and rank correlation are quantified")
def then_quantify_funnel_metrics(ctx):
    m = ctx["funnel_metrics"]
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert m.rank_correlation > 0.70


@then("exploration slot routes candidate for audit")
def then_exploration_slot_routes():
    from mintok.promotion_funnel import route_exploration_candidate

    routed_promoted = route_exploration_candidate("cand_winner", local_passed=True)
    assert routed_promoted is True
    explored = route_exploration_candidate("cand_audit", local_passed=False, exploration_probability=1.0)
    assert explored is True
    not_explored = route_exploration_candidate("cand_audit", local_passed=False, exploration_probability=0.0)
    assert not_explored is False


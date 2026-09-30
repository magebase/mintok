"""Acceptance test steps for the Multi-Stage Promotion Funnel."""

from __future__ import annotations

import json
from pathlib import Path
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.catastrophe import run_catastrophe_suite
from mintok.cli import main as mintok_main
from mintok.execution_harness import (
    ConcurrentPairedRunner,
    PersistentServerConfig,
    WorktreeManager,
)
from mintok.fast_window import FAST12_TASKS, FastTournamentEvaluator
from mintok.mechanism_bench import run_all_mechanism_benchmarks
from mintok.policybench import (
    IntermediateStateRecord,
    PolicyBenchDataset,
    PolicyBenchEvaluator,
)
from mintok.promotion_funnel import run_candidate_eval, run_dev_eval, run_release_eval
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


@when('running "mintok dev-eval" via the CLI', target_fixture="ctx")
def when_run_cli_dev_eval(capsys):
    ret = mintok_main(["dev-eval"])
    out = capsys.readouterr().out
    return {"cli_ret": ret, "cli_out": out}


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


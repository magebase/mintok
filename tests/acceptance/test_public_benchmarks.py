from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.public_bench import (
    PublicBenchmarkReport,
    PublicBenchmarkTask,
    PublicRunRecord,
    assert_workspace_isolation,
    evaluate_paired_public_runs,
    freeze_benchmark_window,
    normalize_swe_bench_pro_task,
    normalize_swe_rebench_task,
    normalize_terminal_bench_task,
    verify_window_fingerprint,
)

scenarios("public_benchmarks.feature")


@given(parsers.parse('a raw SWE-rebench task dictionary with instance id "{instance_id}"'))
def raw_swe_rebench_task(ctx: SimpleNamespace, instance_id: str) -> None:
    ctx.raw = {
        "instance_id": instance_id,
        "repo": "0b01001001/spectree",
        "base_commit": "a091fab020ac26548250c907bae0855273a98778",
        "problem_statement": "Description for query parameters cannot show in swagger UI",
        "patch": "diff --git a/setup.py b/setup.py\n...",
        "test_patch": "diff --git a/tests/test_utils.py ...",
        "install_config": {"test_cmd": "pytest -q"},
        "FAIL_TO_PASS": ["tests/test_utils.py::test_parse_params"],
        "PASS_TO_PASS": ["tests/test_utils.py::test_comments"],
    }


@given(parsers.parse('a raw Terminal-Bench 2.0 task dictionary with instance id "{instance_id}"'))
def raw_terminal_bench_task(ctx: SimpleNamespace, instance_id: str) -> None:
    ctx.raw = {
        "instance_id": instance_id,
        "repo": "terminal-bench/env",
        "base_commit": "main",
        "problem_statement": "Fix syntax error in nginx.conf reverse proxy block",
        "test_cmd": "bash -c './test.sh'",
        "benchmark": "terminal-bench-2.0",
    }


@when("the task is normalized into a public benchmark task")
def normalize_task(ctx: SimpleNamespace) -> None:
    if ctx.raw.get("benchmark") == "terminal-bench-2.0" or "terminal" in ctx.raw.get("instance_id", ""):
        ctx.task = normalize_terminal_bench_task(ctx.raw)
    elif ctx.raw.get("benchmark") == "swe-bench-pro-v2" or "pallets" in ctx.raw.get("repo", ""):
        ctx.task = normalize_swe_bench_pro_task(ctx.raw)
    else:
        ctx.task = normalize_swe_rebench_task(ctx.raw)


@then(parsers.parse('the task has repository "{repo}"'))
def check_task_repo(ctx: SimpleNamespace, repo: str) -> None:
    assert ctx.task.repo == repo


@then(parsers.parse('the benchmark identifier is "{benchmark}"'))
def check_task_benchmark(ctx: SimpleNamespace, benchmark: str) -> None:
    assert ctx.task.benchmark == benchmark


@then("the problem statement matches the issue description")
def check_problem_statement(ctx: SimpleNamespace) -> None:
    assert ctx.task.problem_statement == ctx.raw["problem_statement"]


@then("the reference solution patch is captured in the gold metadata")
def check_patch_captured(ctx: SimpleNamespace) -> None:
    assert ctx.task.patch == ctx.raw["patch"]


@given(parsers.parse('a raw SWE-Bench Pro V2 task dictionary with instance id "{instance_id}"'))
def raw_swe_bench_pro_task(ctx: SimpleNamespace, instance_id: str) -> None:
    ctx.raw = {
        "instance_id": instance_id,
        "repo": "pallets/flask",
        "base_commit": "3c7b2a9",
        "problem_statement": "Fix url routing when trailing slash present",
        "patch": "diff --git a/src/flask/app.py ...",
        "test_patch": "diff --git a/tests/test_basic.py ...",
        "test_cmd": "pytest -q",
        "FAIL_TO_PASS": ["tests/test_basic.py::test_request"],
        "PASS_TO_PASS": [],
    }


@then(parsers.parse('the fail-to-pass list contains "{test_name}"'))
def check_fail_to_pass(ctx: SimpleNamespace, test_name: str) -> None:
    assert test_name in ctx.task.fail_to_pass


@given("a public benchmark window containing 3 tasks")
def window_tasks(ctx: SimpleNamespace) -> None:
    ctx.tasks = [
        PublicBenchmarkTask(
            instance_id=f"repo__task-{i}",
            repo=f"org/repo-{i}",
            base_commit=f"commit-{i}",
            problem_statement=f"Problem {i}",
            benchmark="swe-rebench",
        )
        for i in range(1, 4)
    ]


@when(parsers.parse('the window is frozen with window name "{window_name}"'))
def freeze_window(ctx: SimpleNamespace, window_name: str) -> None:
    ctx.window = freeze_benchmark_window(ctx.tasks, window_name)


@then("the window manifest contains an immutable SHA-256 fingerprint")
def verify_manifest_fingerprint(ctx: SimpleNamespace) -> None:
    assert "window_fingerprint" in ctx.window
    assert len(ctx.window["window_fingerprint"]) == 64
    assert verify_window_fingerprint(ctx.window) is True


@then("tampering with any task prompt or commit fails fingerprint verification")
def verify_tampering_detected(ctx: SimpleNamespace) -> None:
    tampered = dict(ctx.window)
    tampered_tasks = list(ctx.window["tasks"])
    tampered_tasks[0] = dict(tampered_tasks[0], problem_statement="TAMPERED")
    tampered["tasks"] = tampered_tasks
    assert verify_window_fingerprint(tampered) is False


@given(parsers.parse('a public task with repository "{repo}" and gold patch "{gold_patch}"'))
def public_task_with_patch(ctx: SimpleNamespace, repo: str, gold_patch: str) -> None:
    ctx.task = PublicBenchmarkTask(
        instance_id="task-01",
        repo=repo,
        base_commit="abc123",
        problem_statement="A public issue",
        benchmark="swe-rebench",
        patch=gold_patch,
    )


@when("the task workspace is prepared at an isolated destination")
def prepare_workspace_isolation(ctx: SimpleNamespace) -> None:
    tmp = tempfile.mkdtemp()
    ctx.dest = Path(tmp)
    # Simulate clean tree without solution artifacts
    (ctx.dest / "src").mkdir(parents=True)
    (ctx.dest / "src" / "main.py").write_text("def hello(): return 1")


@then("no gold patch or solution file exists within the workspace")
def verify_no_solutions(ctx: SimpleNamespace) -> None:
    assert_workspace_isolation(ctx.dest, ctx.task)


@then("agent tools reject paths escaping the workspace root")
def verify_agent_tools_reject_escape(ctx: SimpleNamespace) -> None:
    from mintok.abi import AgentABI
    abi = AgentABI(ctx.dest)
    with pytest.raises(ValueError):
        abi._safe_path("../secret.json")


@given(parsers.parse("10 paired public benchmark runs where control uses {ctrl_tok:d} provider tokens and MinTok uses {min_tok:d}"))
def setup_paired_runs(ctx: SimpleNamespace, ctrl_tok: int, min_tok: int) -> None:
    ctx.control_runs = []
    ctx.mintok_runs = []
    for i in range(10):
        solved = i < 8  # 80% solve rate
        ctx.control_runs.append(
            PublicRunRecord(
                task_id=f"t{i}",
                arm="control",
                solved=solved,
                provider_tokens=ctrl_tok // 10,
                input_tokens=ctrl_tok // 10,
                output_tokens=100,
                turns=5,
            )
        )
        ctx.mintok_runs.append(
            PublicRunRecord(
                task_id=f"t{i}",
                arm="mintok",
                solved=solved,
                provider_tokens=min_tok // 10,
                input_tokens=min_tok // 10,
                output_tokens=50,
                turns=3,
            )
        )


@when("the public benchmark report is generated")
def generate_public_report(ctx: SimpleNamespace) -> None:
    ctx.report = evaluate_paired_public_runs(ctx.control_runs, ctx.mintok_runs)


@then("the control solve rate is 80%")
def check_ctrl_solve(ctx: SimpleNamespace) -> None:
    assert ctx.report.control_solve_rate == 0.8


@then("the MinTok solve rate is 80%")
def check_mintok_solve(ctx: SimpleNamespace) -> None:
    assert ctx.report.mintok_solve_rate == 0.8


@then(parsers.parse("the primary efficiency multiplier is {expected}"))
def check_efficiency_multiplier(ctx: SimpleNamespace, expected: str) -> None:
    exp_val = float(expected.rstrip("x"))
    assert abs(ctx.report.efficiency_multiplier - exp_val) < 0.05


@then(parsers.parse('the report status is "{status}"'))
def check_report_status(ctx: SimpleNamespace, status: str) -> None:
    assert ctx.report.gate_verdict == status


@then("the paired geometric mean savings is reported")
def check_geomean_reported(ctx: SimpleNamespace) -> None:
    assert ctx.report.both_solved_geomean > 1.0

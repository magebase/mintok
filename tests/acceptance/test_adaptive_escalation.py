from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "e2e"))
import agent_cli  # noqa: E402

from mintok.escalation import (
    EscalationController,
    EscalationLevel,
    TrajectoryEvent,
    compute_yield,
    extract_traceback_target,
)

scenarios("adaptive_escalation.feature")


@given("an empty trajectory")
def empty_trajectory(ctx: SimpleNamespace) -> None:
    ctx.controller = EscalationController()


@when(parsers.parse('a tool "{tool}" fails with "{output}"'))
def tool_fails(ctx: SimpleNamespace, tool: str, output: str) -> None:
    ctx.controller.record_event(TrajectoryEvent(tool=tool, exit_code=1, output=output))


@when(parsers.parse('a tool "{tool}" succeeds with {tokens:d} tokens and output "{output}"'))
def tool_succeeds(ctx: SimpleNamespace, tool: str, tokens: int, output: str) -> None:
    unescaped_output = output.encode("utf-8").decode("unicode_escape")
    ctx.controller.record_event(TrajectoryEvent(tool=tool, exit_code=0, output=unescaped_output, tokens=tokens))


@then(parsers.parse("the trajectory state has {count:d} missing file read"))
@then(parsers.parse("the trajectory state has {count:d} missing file reads"))
def has_missing_file_reads(ctx: SimpleNamespace, count: int) -> None:
    assert ctx.controller.features.missing_file_reads == count


@then(parsers.parse("the trajectory state has {count:d} rejected patch"))
@then(parsers.parse("the trajectory state has {count:d} rejected patches"))
def has_rejected_patches(ctx: SimpleNamespace, count: int) -> None:
    assert ctx.controller.features.rejected_patches == count


@then(parsers.parse("the turns elapsed is {count:d}"))
def turns_elapsed(ctx: SimpleNamespace, count: int) -> None:
    assert ctx.controller.features.turns == count


@when("a test failure occurs with output:")
def step_test_failure(ctx: SimpleNamespace, docstring: str) -> None:
    ctx.controller.record_event(TrajectoryEvent(tool="suite", exit_code=1, output=docstring))


@when("another test failure occurs with output:")
def step_another_test_failure(ctx: SimpleNamespace, docstring: str) -> None:
    ctx.controller.record_event(TrajectoryEvent(tool="suite", exit_code=1, output=docstring))


@then(parsers.parse("the consecutive test failure count is {count:d}"))
def consecutive_test_failures(ctx: SimpleNamespace, count: int) -> None:
    assert ctx.controller.features.consecutive_identical_test_failures == count


@then("stagnation is detected")
def stagnation_detected(ctx: SimpleNamespace) -> None:
    assert ctx.controller.features.stagnated is True


@then("stagnation is not detected")
def stagnation_not_detected(ctx: SimpleNamespace) -> None:
    assert ctx.controller.features.stagnated is False


@given("a test failure traceback:")
def step_failure_traceback(ctx: SimpleNamespace, docstring: str) -> None:
    ctx.traceback = docstring


@when("the traceback is parsed")
def parse_traceback(ctx: SimpleNamespace) -> None:
    target = extract_traceback_target(ctx.traceback)
    ctx.target_str = f"{target[0]}:{target[1]}" if target else None


@then(parsers.parse('the failing target is "{expected}"'))
def check_failing_target(ctx: SimpleNamespace, expected: str) -> None:
    assert ctx.target_str == expected


@given(parsers.parse("an escalation controller at level {level:d}"))
def controller_at_level(ctx: SimpleNamespace, level: int) -> None:
    ctx.controller = EscalationController(initial_level=EscalationLevel(level))


@when(parsers.parse("{count:d} missing file reads occur"))
def missing_file_reads_occur(ctx: SimpleNamespace, count: int) -> None:
    for _ in range(count):
        ctx.controller.record_event(TrajectoryEvent(tool="read", exit_code=1, output="error: no such file foo.py"))


@then(parsers.parse("the escalation controller level is at least {level:d}"))
def level_at_least(ctx: SimpleNamespace, level: int) -> None:
    assert ctx.controller.level >= level


@then(parsers.parse('the active tools include "{tool}"'))
def active_tools_include(ctx: SimpleNamespace, tool: str) -> None:
    assert tool in ctx.controller.active_tools()


@when(parsers.parse("{count:d} consecutive identical test failures occur"))
def consecutive_identical_failures_occur(ctx: SimpleNamespace, count: int) -> None:
    for _ in range(count):
        ctx.controller.record_event(TrajectoryEvent(tool="suite", exit_code=1, output="FAILED tests/t.py - AssertionError: 1 != 2"))


@when(parsers.parse("{count:d} consecutive read queries occur without edits"))
def consecutive_reads_without_edits_occur(ctx: SimpleNamespace, count: int) -> None:
    for _ in range(count):
        ctx.controller.record_event(TrajectoryEvent(tool="read", exit_code=0, output="def foo(): pass"))


@when(parsers.parse("{count:d} empty slice queries occur"))
def empty_slice_queries_occur(ctx: SimpleNamespace, count: int) -> None:
    for _ in range(count):
        ctx.controller.record_event(TrajectoryEvent(tool="slice", exit_code=0, output="0 candidates found"))


@then(parsers.parse("the escalation controller level is {level:d}"))
def level_is(ctx: SimpleNamespace, level: int) -> None:
    assert ctx.controller.level == level


@given(parsers.parse("{solved:d} solved tasks out of {attempts:d} attempts"))
def solved_out_of_attempts(ctx: SimpleNamespace, solved: int, attempts: int) -> None:
    ctx.solved = solved
    ctx.attempts = attempts


@given(parsers.parse("a total spend of {tokens:d} provider tokens"))
def total_tokens(ctx: SimpleNamespace, tokens: int) -> None:
    ctx.tokens = tokens


@when("economic efficiency is computed")
def compute_efficiency(ctx: SimpleNamespace) -> None:
    ctx.yield_metric = compute_yield(ctx.solved, ctx.tokens)


@then(parsers.parse("the yield is {expected:f} solves per million tokens"))
def check_yield(ctx: SimpleNamespace, expected: float) -> None:
    assert abs(ctx.yield_metric - expected) < 1e-2


@given(parsers.parse('an empty trajectory log "{log_name}"'))
def empty_log_file(ctx: SimpleNamespace, repo: Path, log_name: str) -> None:
    ctx.log_path = repo / log_name
    ctx.log_path.unlink(missing_ok=True)


@given(parsers.parse('a trajectory log "{log_name}" with 2 identical test failures'))
def log_with_two_test_failures(ctx: SimpleNamespace, repo: Path, log_name: str) -> None:
    ctx.log_path = repo / log_name
    lines = [
        {"tool": "suite", "output": "FAILED test.py - AssertionError: fail", "exit_code": 1},
        {"tool": "suite", "output": "FAILED test.py - AssertionError: fail", "exit_code": 1},
    ]
    ctx.log_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@when(parsers.parse('the agent CLI executes tool "{tool}" with args "{tool_args}" under policy "{policy}"'))
def execute_agent_cli(ctx: SimpleNamespace, repo: Path, tool: str, tool_args: str, policy: str, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--root", str(repo), "--log", str(ctx.log_path), "--policy", policy, tool, tool_args]
    try:
        ctx.agent_exit_code = agent_cli.main(argv)
    except SystemExit as exc:
        ctx.agent_exit_code = exc.code
    ctx.agent_output = capsys.readouterr().out


@then(parsers.parse("the agent CLI exit code is {code:d}"))
def check_agent_cli_exit(ctx: SimpleNamespace, code: int) -> None:
    assert ctx.agent_exit_code == code, f"Expected {code} got {ctx.agent_exit_code}: {ctx.agent_output}"


@then(parsers.parse('the agent CLI output includes "{text}"'))
def check_agent_cli_output(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.agent_output, ctx.agent_output

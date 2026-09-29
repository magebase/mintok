"""Acceptance steps for MinTok 3.0 runtime inference optimizer."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "e2e"))
import agent_cli

scenarios("v3_optimizer.feature")


@given(parsers.parse('an empty trajectory log "{log_name}"'))
def given_empty_log(ctx: SimpleNamespace, repo: Path, log_name: str) -> None:
    ctx.log_path = repo / log_name
    ctx.log_path.unlink(missing_ok=True)
    obs_dir = repo / "observations"
    if obs_dir.exists():
        for p in obs_dir.glob("*.json"):
            p.unlink()


@given(parsers.parse('a repository with a failing function "{func}" in "{fpath}"'))
def given_repo_with_func(repo: Path, func: str, fpath: str):
    p = repo / fpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"def {func}(x):\n"
        f"    if x <= 0:\n"
        f"        raise ValueError('x must be positive')\n"
        f"    return x * 2\n",
        encoding="utf-8",
    )


@when(parsers.parse('the agent CLI executes tool "{tool}" with args "{tool_args}" under policy "{policy}"'))
def execute_cli_tool(ctx: SimpleNamespace, repo: Path, tool: str, tool_args: str, policy: str, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--root", str(repo), "--log", str(ctx.log_path), "--policy", policy, tool, tool_args]
    try:
        ctx.exit_code = agent_cli.main(argv)
    except SystemExit as exc:
        ctx.exit_code = exc.code
    ctx.output = capsys.readouterr().out


@when(parsers.parse('the latest observation handle is expanded with filter "{filter_str}" under policy "{policy}"'))
def expand_latest_obs(ctx: SimpleNamespace, repo: Path, filter_str: str, policy: str, capsys: pytest.CaptureFixture[str]) -> None:
    m = re.search(r"obs:[a-f0-9]+", ctx.output)
    assert m is not None, f"No obs handle found in {ctx.output}"
    handle = m.group(0)
    argv = ["--root", str(repo), "--log", str(ctx.log_path), "--policy", policy, "expand", handle, "--filter", filter_str]
    try:
        ctx.exit_code = agent_cli.main(argv)
    except SystemExit as exc:
        ctx.exit_code = exc.code
    ctx.output = capsys.readouterr().out


@when(parsers.parse('the agent CLI executes macro "{macro}" on "{target}" under policy "{policy}"'))
def execute_macro(ctx: SimpleNamespace, repo: Path, macro: str, target: str, policy: str, capsys: pytest.CaptureFixture[str]) -> None:
    tb = f'File "{target.split(":")[0]}", line {target.split(":")[1]}, in calculate'
    argv = ["--root", str(repo), "--log", str(ctx.log_path), "--policy", policy, macro, "--traceback", tb]
    try:
        ctx.exit_code = agent_cli.main(argv)
    except SystemExit as exc:
        ctx.exit_code = exc.code
    ctx.output = capsys.readouterr().out


@then(parsers.parse("the agent CLI exit code is {code:d}"))
def check_exit_code(ctx: SimpleNamespace, code: int) -> None:
    assert ctx.exit_code == code, f"Expected {code} got {ctx.exit_code}: {ctx.output}"


@then(parsers.parse('the agent CLI output includes "{text}"'))
def check_output_includes(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.output, f"Expected '{text}' in:\n{ctx.output}"

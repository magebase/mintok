from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from mintok.cli import main

scenarios("token_bench.feature")


@given("a copy of the repository is made")
def copy_repo(ctx: SimpleNamespace, repo: Path) -> None:
    ctx.copy_root = repo.parent / "repo-copy"
    shutil.copytree(repo, ctx.copy_root)


@given(parsers.parse('file "{name}" in the copy is replaced with:'))
def replace_file_in_copy(ctx: SimpleNamespace, name: str, docstring: str) -> None:
    (ctx.copy_root / name).write_text(textwrap.dedent(docstring).strip("\n") + "\n")


@when(parsers.parse('the token benchmark runs alone on "{root}" with JSON output'))
def bench_json(ctx: SimpleNamespace, repo: Path, root: str, capsys: pytest.CaptureFixture[str]) -> None:
    ctx.exit_code = main(["bench", root.format(repo=repo), "--format", "json"])
    ctx.bench = json.loads(capsys.readouterr().out)


@when(parsers.parse('the token benchmark runs alone on "{root}" as text'))
def bench_text(ctx: SimpleNamespace, repo: Path, root: str, capsys: pytest.CaptureFixture[str]) -> None:
    ctx.exit_code = main(["bench", root.format(repo=repo)])
    ctx.cli_output = capsys.readouterr().out


@when(parsers.parse('the token benchmark runs with a second version on "{root}" against "{other}" with JSON output'))
def bench_vs(
    ctx: SimpleNamespace, repo: Path, root: str, other: str, capsys: pytest.CaptureFixture[str]
) -> None:
    old = root.format(repo=repo)
    new = other.format(repo=repo, copy=getattr(ctx, "copy_root", ""))
    ctx.exit_code = main(["bench", old, "--vs", new, "--format", "json"])
    ctx.bench = json.loads(capsys.readouterr().out)


@then("the benchmark JSON reports baseline tokens above treatment tokens")
def bench_reduces(ctx: SimpleNamespace) -> None:
    assert ctx.bench["baseline_tokens"] > ctx.bench["treatment_tokens"], ctx.bench


@then("every benchmark task costs fewer treatment tokens than baseline tokens")
def bench_tasks_reduce(ctx: SimpleNamespace) -> None:
    worse = [t for t in ctx.bench["tasks"] if t["treatment_tokens"] >= t["baseline_tokens"]]
    assert not worse, worse


@then(parsers.parse('the benchmark JSON reports a reduction ratio of at least "{minimum}"'))
def bench_ratio(ctx: SimpleNamespace, minimum: str) -> None:
    assert ctx.bench["reduction_ratio"] >= float(minimum), ctx.bench["reduction_ratio"]


@then(parsers.parse('the benchmark JSON reports a strong-tooling reduction ratio of at least "{minimum}"'))
def bench_strong_ratio(ctx: SimpleNamespace, minimum: str) -> None:
    assert ctx.bench["strong_reduction_ratio"] >= float(minimum), ctx.bench["strong_reduction_ratio"]


@then(parsers.parse('the benchmark JSON includes a "{task_class}" task'))
def bench_task_class(ctx: SimpleNamespace, task_class: str) -> None:
    assert any(t["task_class"] == task_class for t in ctx.bench["tasks"]), ctx.bench["tasks"]

from __future__ import annotations

import shlex
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, then, when

from mintok.cli import main
from mintok.compiler import compile_repository
from tests.acceptance.helpers import split_list


@pytest.fixture
def ctx() -> SimpleNamespace:
    return SimpleNamespace(ir=None, previous_ir=None)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def _write(repo: Path, name: str, docstring: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(docstring).strip("\n") + "\n")


@given(parsers.parse('a repository file "{name}":'))
def repository_file(repo: Path, name: str, docstring: str) -> None:
    _write(repo, name, docstring)


@when(parsers.parse('file "{name}" is replaced with:'))
def replace_file(repo: Path, name: str, docstring: str) -> None:
    _write(repo, name, docstring)


@given("the repository is compiled")
@when("the repository is compiled")
def compile_repo(ctx: SimpleNamespace, repo: Path) -> None:
    ctx.ir = compile_repository(repo)
    assert not ctx.ir.diagnostics, ctx.ir.diagnostics


@when("the repository is compiled again")
def compile_again(ctx: SimpleNamespace, repo: Path) -> None:
    ctx.previous_ir = ctx.ir
    ctx.ir = compile_repository(repo)
    assert not ctx.ir.diagnostics, ctx.ir.diagnostics


# Steps shared across feature files live here (pytest-bdd scopes module-level
# step definitions to their own module).


@then(parsers.parse('symbol "{sid}" calls "{target}" with confidence {confidence:f}'))
def calls_with_confidence(ctx: SimpleNamespace, sid: str, target: str, confidence: float) -> None:
    calls = {f.object: f.confidence for f in ctx.ir.facts_for(sid, "calls")}
    assert calls.get(target) == confidence, calls


@when(parsers.parse('I run the CLI with "{args}"'))
def run_cli(ctx: SimpleNamespace, repo: Path, args: str, capsys: pytest.CaptureFixture[str]) -> None:
    ctx.exit_code = main(shlex.split(args.format(repo=repo)))
    ctx.cli_output = capsys.readouterr().out


@then(parsers.parse("the CLI exits with code {code:d}"))
def cli_exit(ctx: SimpleNamespace, code: int) -> None:
    assert ctx.exit_code == code


@then(parsers.parse('the CLI output includes "{text}"'))
def cli_output(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.cli_output, ctx.cli_output

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pytest_bdd import parsers, scenarios, then, when

from mintok.abi import TOOL_SURFACE, AgentABI, ChangeRejected, tool_surface_tokens
from mintok.tokens import estimate_tokens
from tests.acceptance.helpers import split_list

scenarios("agent_abi.feature")


@then(parsers.parse('the agent tool surface is exactly "{names}"'))
def tool_names(names: str) -> None:
    assert [t["name"] for t in TOOL_SURFACE] == split_list(names)


@then(parsers.parse("the agent tool surface costs at most {limit:d} tokens"))
def tool_cost(limit: int) -> None:
    assert tool_surface_tokens() <= limit


@when(parsers.parse('the agent queries symbol "{sid}"'))
def query_symbol(ctx: SimpleNamespace, repo: Path, sid: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    ctx.answer = ctx.abi.query("symbol", sid)


@then(parsers.parse('the answer includes "{text}"'))
def answer_includes(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.answer, ctx.answer


@then(parsers.parse('the answer does not include "{text}"'))
def answer_excludes(ctx: SimpleNamespace, text: str) -> None:
    assert text not in ctx.answer, ctx.answer


@then(parsers.parse('the answer costs fewer tokens than the source of "{sid}"'))
def answer_cheaper(ctx: SimpleNamespace, sid: str) -> None:
    assert estimate_tokens(ctx.answer) < estimate_tokens(ctx.abi.source_of(sid))


@when(parsers.parse('the agent changes "{sid}" to:'))
def agent_change(ctx: SimpleNamespace, repo: Path, sid: str, docstring: str) -> None:
    ctx.original = {p.name: p.read_text() for p in repo.glob("*.py")}
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.change = ctx.abi.change(sid, docstring)
        ctx.change_error = None
    except ChangeRejected as exc:
        ctx.change = None
        ctx.change_error = str(exc)


@then("the change is accepted")
def change_accepted(ctx: SimpleNamespace) -> None:
    assert ctx.change_error is None, ctx.change_error


@then("the change reports the interface as unchanged")
def interface_unchanged(ctx: SimpleNamespace) -> None:
    assert ctx.change.interface_changed is False


@then(parsers.parse('the change is rejected with "{text}"'))
def change_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.change_error is not None and text in ctx.change_error, ctx.change_error


@then(parsers.parse('file "{name}" still defines "{sid}"'))
def still_defines(ctx: SimpleNamespace, name: str, sid: str) -> None:
    assert sid in ctx.abi.ir.symbols
    assert ctx.abi.ir.symbols[sid].source.path == name


@then(parsers.parse('file "{name}" is unmodified'))
def unmodified(ctx: SimpleNamespace, repo: Path, name: str) -> None:
    assert (repo / name).read_text() == ctx.original[name]


@when("the agent verifies with a command that prints 50 lines and succeeds")
def agent_verify(ctx: SimpleNamespace, repo: Path) -> None:
    abi = AgentABI(repo, ctx.ir)
    ctx.verify = abi.verify([sys.executable, "-c", "for i in range(50): print(i)"])


@then("verification passed")
def verification_passed(ctx: SimpleNamespace) -> None:
    assert ctx.verify.passed and ctx.verify.exit_code == 0


@then(parsers.parse("the verification output has at most {n:d} lines"))
def verification_tail(ctx: SimpleNamespace, n: int) -> None:
    assert len(ctx.verify.tail.splitlines()) <= n


@then(parsers.parse('the file "{name}" is IR version "{version}" containing symbol "{sid}"'))
def ir_file(repo: Path, name: str, version: str, sid: str) -> None:
    data = json.loads((repo / name).read_text())
    assert data["version"] == version
    assert sid in {s["id"] for s in data["symbols"]}

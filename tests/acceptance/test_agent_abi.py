from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pytest_bdd import parsers, scenarios, then, when

from mintok.abi import TOOL_SURFACE, AgentABI, AddResult, ChangeRejected, tool_surface_tokens
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


@when(parsers.parse('the agent batch-queries "{spec}"'))
def batch_query(ctx: SimpleNamespace, repo: Path, spec: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.answer = ctx.abi.query("batch", spec)
        ctx.batch_error = None
    except (KeyError, ValueError) as exc:
        ctx.answer = ""
        ctx.batch_error = str(exc)


@then(parsers.parse('the batch is rejected with "{text}"'))
def batch_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.batch_error is not None and text in ctx.batch_error, ctx.batch_error


@when(parsers.parse('the agent renames "{sid}" to "{new_name}"'))
def codemod_rename(ctx: SimpleNamespace, repo: Path, sid: str, new_name: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.codemod = ctx.abi.codemod("rename", target=sid, to=new_name)
        ctx.codemod_error = None
    except ChangeRejected as exc:
        ctx.codemod = None
        ctx.codemod_error = str(exc)


@when(parsers.parse('the agent adds parameter "{name}" with default "{default}" to "{sid}"'))
def codemod_add_parameter(
    ctx: SimpleNamespace, repo: Path, name: str, default: str, sid: str
) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.codemod = ctx.abi.codemod("add_parameter", target=sid, name=name, default=default)
        ctx.codemod_error = None
    except ChangeRejected as exc:
        ctx.codemod = None
        ctx.codemod_error = str(exc)


@when(parsers.parse('the agent adds the import "{statement}" to "{path}"'))
def codemod_add_import(ctx: SimpleNamespace, repo: Path, statement: str, path: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.codemod = ctx.abi.codemod("add_import", file=path, statement=statement)
        ctx.codemod_error = None
    except ChangeRejected as exc:
        ctx.codemod = None
        ctx.codemod_error = str(exc)


@then("the codemod is accepted")
def codemod_accepted(ctx: SimpleNamespace) -> None:
    assert ctx.codemod_error is None, ctx.codemod_error


@then(parsers.parse('the codemod is rejected with "{text}"'))
def codemod_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.codemod_error is not None and text in ctx.codemod_error, ctx.codemod_error


@then(parsers.parse('the symbol "{sid}" is in the IR'))
def symbol_in_ir(ctx: SimpleNamespace, sid: str) -> None:
    assert sid in ctx.abi.ir.symbols, sorted(ctx.abi.ir.symbols)[:10]


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


@then("the change reports the interface as changed")
def interface_changed(ctx: SimpleNamespace) -> None:
    assert ctx.change.interface_changed is True


@then(parsers.parse('the file "{name}" includes the line "{text}"'))
def file_includes_line(ctx: SimpleNamespace, repo: Path, name: str, text: str) -> None:
    content = (repo / name).read_text()
    assert any(text in line for line in content.splitlines()), content


@then(parsers.parse('the change is rejected with "{text}"'))
def change_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.change_error is not None and text in ctx.change_error, ctx.change_error


@then(parsers.parse('file "{name}" still defines "{sid}"'))
def still_defines(ctx: SimpleNamespace, name: str, sid: str) -> None:
    assert sid in ctx.abi.ir.symbols
    assert ctx.abi.ir.symbols[sid].source.path == name


@when(parsers.parse('the agent adds "{sid}" to "{path}" with source:'))
def agent_add(ctx: SimpleNamespace, repo: Path, sid: str, path: str, docstring: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    _run_add(ctx, repo, sid, path, docstring, imports=None)


@when(parsers.parse('the agent adds "{sid}" to "{path}" with imports "{imports}" and source:'))
def agent_add_imports(
    ctx: SimpleNamespace, repo: Path, sid: str, path: str, imports: str, docstring: str
) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    _run_add(ctx, repo, sid, path, docstring, imports=imports)


def _run_add(
    ctx: SimpleNamespace, repo: Path, sid: str, path: str, source: str, imports: str | None
) -> None:
    ctx.existed_before = (repo / path).exists()
    try:
        ctx.add_result: AddResult | None = ctx.abi.add(
            sid, source, path, imports=split_list(imports) if imports else ()
        )
        ctx.add_error = None
    except ChangeRejected as exc:
        ctx.add_result = None
        ctx.add_error = str(exc)


@then("the addition is accepted")
def add_accepted(ctx: SimpleNamespace) -> None:
    assert ctx.add_error is None, ctx.add_error


@then("the addition created no file")
def add_no_new_file(ctx: SimpleNamespace) -> None:
    assert ctx.add_result.created_file is False


@then(parsers.parse('the addition created the file "{name}"'))
def add_created_file(ctx: SimpleNamespace, name: str) -> None:
    assert ctx.add_result.created_file is True


@then(parsers.parse('the addition is rejected with "{text}"'))
def add_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.add_error is not None and text in ctx.add_error, ctx.add_error


@then(parsers.parse('the line "{text}" comes first in "{name}"'))
def line_comes_first(ctx: SimpleNamespace, text: str, name: str, repo: Path) -> None:
    lines = (repo / name).read_text().splitlines()
    assert text in lines and lines.index(text) == 0, lines[:3]


@when(parsers.parse('the agent inspects "{sid}"'))
def agent_inspect(ctx: SimpleNamespace, repo: Path, sid: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    ctx.answer = ctx.abi.inspect(sid)


@when(parsers.parse('the agent requests a task packet for "{description}"'))
def agent_packet(ctx: SimpleNamespace, repo: Path, description: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.packet = ctx.abi.task_packet(description)
        ctx.packet_error = None
    except ChangeRejected as exc:
        ctx.packet = None
        ctx.packet_error = str(exc)


@then(parsers.parse('the packet names "{sid}"'))
def packet_names(ctx: SimpleNamespace, sid: str) -> None:
    assert ctx.packet_error is None, ctx.packet_error
    assert f"target {sid}" in ctx.packet, ctx.packet


@then(parsers.parse('the packet includes "{section}"'))
def packet_includes(ctx: SimpleNamespace, section: str) -> None:
    assert section in ctx.packet, ctx.packet


@when(parsers.parse('the agent patches "{name}" lines {start:d} to {end:d} with:'))
def agent_patch(
    ctx: SimpleNamespace, repo: Path, name: str, start: int, end: int, docstring: str
) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.patch_result = ctx.abi.patch(name, start, end, docstring)
        ctx.patch_error = None
    except ChangeRejected as exc:
        ctx.patch_result = None
        ctx.patch_error = str(exc)


@then("the patch is accepted")
def patch_accepted(ctx: SimpleNamespace) -> None:
    assert ctx.patch_error is None, ctx.patch_error


@then(parsers.parse('the patch is rejected with "{text}"'))
def patch_rejected(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.patch_error is not None and text in ctx.patch_error, ctx.patch_error


@when(parsers.parse('the agent removes "{sid}"'))
def agent_remove(ctx: SimpleNamespace, repo: Path, sid: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    try:
        ctx.remove_result = ctx.abi.remove(sid)
        ctx.remove_error = None
    except ChangeRejected as exc:
        ctx.remove_result = None
        ctx.remove_error = str(exc)


@then("the removal is accepted")
def removal_accepted(ctx: SimpleNamespace) -> None:
    assert ctx.remove_error is None, ctx.remove_error


@then("the removal reports no remaining references")
def removal_no_dependents(ctx: SimpleNamespace) -> None:
    assert ctx.remove_result.dependents == (), ctx.remove_result.dependents


@then(parsers.parse('the removal reports remaining references from "{sids}"'))
def removal_dependents(ctx: SimpleNamespace, sids: str) -> None:
    assert sorted(ctx.remove_result.dependents) == sorted(split_list(sids)), ctx.remove_result.dependents


@then(parsers.parse('the symbol "{sid}" is gone from the IR'))
def symbol_gone(ctx: SimpleNamespace, sid: str) -> None:
    assert sid not in ctx.abi.ir.symbols


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

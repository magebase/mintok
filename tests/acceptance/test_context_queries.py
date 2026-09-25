from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import parsers, scenarios, then, when

from mintok.abi import AgentABI
from mintok.tokens import estimate_tokens

scenarios("context_queries.feature")


@when(parsers.parse('the agent finds symbols matching "{pattern}"'))
def agent_finds(ctx: SimpleNamespace, repo: Path, pattern: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    ctx.answer = ctx.abi.query("find", pattern)


@then(parsers.parse('the find answer lists symbols "{sids}"'))
def find_lists(ctx: SimpleNamespace, sids: str) -> None:
    listed = [line.split(" ", 1)[0] for line in ctx.answer.splitlines()]
    assert listed == [s.strip() for s in sids.split(",")], listed


@then(parsers.parse('the find answer costs fewer tokens than the source of "{sid}"'))
def find_cheaper(ctx: SimpleNamespace, sid: str) -> None:
    source = ctx.abi.source_of(sid)
    assert estimate_tokens(ctx.answer) < estimate_tokens(source), (ctx.answer, source)


@when(parsers.parse('the agent queries writers of attribute "{attribute}"'))
def agent_writers(ctx: SimpleNamespace, repo: Path, attribute: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    ctx.answer = ctx.abi.query("writers", attribute)


@then(parsers.parse('the answer names writer "{text}"'))
def answer_names_writer(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.answer, ctx.answer


@when(parsers.parse('the agent queries summary "{sid}"'))
def agent_summary(ctx: SimpleNamespace, repo: Path, sid: str) -> None:
    ctx.abi = AgentABI(repo, ctx.ir)
    ctx.answer = ctx.abi.query("summary", sid)


@then(parsers.parse('the summary answer includes "{text}"'))
def summary_includes(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.answer, ctx.answer


@then(parsers.parse('the summary answer omits "{text}"'))
def summary_omits(ctx: SimpleNamespace, text: str) -> None:
    assert text not in ctx.answer, ctx.answer

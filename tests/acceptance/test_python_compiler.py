from __future__ import annotations

from types import SimpleNamespace

from pytest_bdd import parsers, scenarios, then

from tests.acceptance.helpers import split_list

scenarios("python_compiler.feature")


@then(parsers.parse('the IR contains symbol "{sid}" of kind "{kind}"'))
def has_symbol(ctx: SimpleNamespace, sid: str, kind: str) -> None:
    assert sid in ctx.ir.symbols, sorted(ctx.ir.symbols)
    assert ctx.ir.symbols[sid].kind == kind


@then(parsers.parse('symbol "{sid}" has signature "{signature}"'))
def has_signature(ctx: SimpleNamespace, sid: str, signature: str) -> None:
    assert ctx.ir.symbols[sid].signature == signature


@then(parsers.parse('symbol "{sid}" has fact "{predicate}" "{obj}"'))
def has_fact(ctx: SimpleNamespace, sid: str, predicate: str, obj: str) -> None:
    objects = [f.object for f in ctx.ir.facts_for(sid, predicate)]
    assert obj in objects, objects


@then(parsers.parse('the callers of "{sid}" are "{callers}"'))
def callers_are(ctx: SimpleNamespace, sid: str, callers: str) -> None:
    assert sorted(f.subject for f in ctx.ir.callers_of(sid)) == sorted(split_list(callers))


@then(parsers.parse('the {which} hash of "{sid}" is {state}'))
def hash_state(ctx: SimpleNamespace, which: str, sid: str, state: str) -> None:
    attr = {"body": "body_hash", "interface": "interface_hash"}[which]
    before = getattr(ctx.previous_ir.symbols[sid], attr)
    after = getattr(ctx.ir.symbols[sid], attr)
    assert state in {"changed", "unchanged"}
    assert (before == after) is (state == "unchanged"), (before, after)

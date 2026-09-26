from __future__ import annotations

from types import SimpleNamespace

from pytest_bdd import parsers, scenarios, then, when

from mintok.abi import AgentABI
from mintok.continuity import LearnedState, file_digests

scenarios("semantic_continuity.feature")


def _learn_symbol(ctx: SimpleNamespace, repo, sid: str) -> None:
    abi = AgentABI(repo, ctx.ir)
    ctx.state = getattr(ctx, "state", None) or LearnedState()
    digests = file_digests(repo)
    ctx.answer = abi.query("symbol", sid)
    ctx.state.observe_symbol(abi.ir.symbols[sid], digests)


def _requery_symbol(ctx: SimpleNamespace, repo, sid: str) -> None:
    digests = file_digests(repo)
    verdict = ctx.state.symbol_verdict(digests, ctx.ir, sid)
    if verdict is not None:
        ctx.answer = verdict
        return
    abi = AgentABI(repo, ctx.ir)
    ctx.answer = abi.query("symbol", sid)
    # a full show is a learn: the agent has now seen the current facts
    ctx.state.observe_symbol(abi.ir.symbols[sid], digests)


@when(parsers.parse('the agent learns symbol "{sid}"'))
def learn_symbol(ctx: SimpleNamespace, repo, sid: str) -> None:
    _learn_symbol(ctx, repo, sid)


@when(parsers.parse('the agent re-queries symbol "{sid}" with unchanged sources'))
def requery_symbol_clean(ctx: SimpleNamespace, repo, sid: str) -> None:
    _requery_symbol(ctx, repo, sid)


@when(parsers.parse('the agent re-queries symbol "{sid}" after the edit'))
def requery_symbol_edited(ctx: SimpleNamespace, repo, sid: str) -> None:
    _requery_symbol(ctx, repo, sid)


@then(parsers.parse('the answer is the verdict "{text}"'))
def answer_is_verdict(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.answer == text, ctx.answer


@then(parsers.parse('the answer shows the full facts for "{sid}"'))
def answer_is_full(ctx: SimpleNamespace, sid: str) -> None:
    assert "unchanged since learn" not in ctx.answer, ctx.answer


@then(parsers.parse('the learned record for "{sid}" is updated'))
def record_updated(ctx: SimpleNamespace, sid: str) -> None:
    rec = ctx.state.symbols[sid]
    assert rec["interface_hash"] == ctx.ir.symbols[sid].interface_hash
    assert rec["seq"] > 1


@when(parsers.parse('the agent learns the callers of "{sid}"'))
def learn_callers(ctx: SimpleNamespace, repo, sid: str) -> None:
    abi = AgentABI(repo, ctx.ir)
    ctx.state = getattr(ctx, "state", None) or LearnedState()
    digests = file_digests(repo)
    callers = abi.get_callers(sid)
    ctx.answer = "\n".join(f"{f.subject} {f.confidence:.2f}" for f in callers)
    ctx.state.observe_relation("callers", sid, digests, len(callers))


@when(parsers.parse('the agent re-queries the callers of "{sid}" with unchanged sources'))
def requery_callers_clean(ctx: SimpleNamespace, repo, sid: str) -> None:
    verdict = ctx.state.relation_verdict("callers", sid, file_digests(repo))
    ctx.answer = verdict if verdict is not None else AgentABI(repo, ctx.ir).get_callers(sid)


@when(parsers.parse('the agent re-queries the callers of "{sid}" after the edit'))
def requery_callers_edited(ctx: SimpleNamespace, repo, sid: str) -> None:
    verdict = ctx.state.relation_verdict("callers", sid, file_digests(repo))
    ctx.answer = verdict if verdict is not None else AgentABI(repo, ctx.ir).get_callers(sid)


@then(parsers.parse('the answer shows the full facts for callers of "{sid}"'))
def answer_is_full_callers(ctx: SimpleNamespace, sid: str) -> None:
    assert "unchanged since learn" not in ctx.answer, ctx.answer


@when(parsers.parse('the agent changes "{sid}" to:'))
def agent_changes(ctx: SimpleNamespace, repo, sid: str, docstring: str) -> None:
    abi = AgentABI(repo, ctx.ir)
    ctx.abi = abi
    abi.change(sid, docstring)
    ctx.delta = ctx.state.write_delta(abi.ir, file_digests(repo))


@then(parsers.parse('the write delta reports "{text}"'))
def delta_reports(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.delta, ctx.delta


@then(parsers.parse('the learned record for "{sid}" carries the new interface'))
def record_carries_new_interface(ctx: SimpleNamespace, sid: str) -> None:
    assert ctx.state.symbols[sid]["interface_hash"] == ctx.abi.ir.symbols[sid].interface_hash


@when("the learned state is serialized and restored")
def serialize_restore(ctx: SimpleNamespace) -> None:
    ctx.state = LearnedState.from_json(ctx.state.to_json())

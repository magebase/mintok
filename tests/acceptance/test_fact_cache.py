from __future__ import annotations

from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.cache import FactCache
from tests.acceptance.helpers import split_list

scenarios("fact_cache.feature")


@given("the compiled facts are cached")
def cache_facts(ctx: SimpleNamespace) -> None:
    ctx.cache = FactCache.from_ir(ctx.ir)


@when("the cache is refreshed from the new compile")
def refresh_cache(ctx: SimpleNamespace) -> None:
    ctx.refresh = ctx.cache.refresh(ctx.ir)


@then(parsers.parse('the invalidated symbols are exactly "{symbols}"'))
def invalidated_exactly(ctx: SimpleNamespace, symbols: str) -> None:
    assert sorted(ctx.refresh.invalidated) == sorted(split_list(symbols))


@then("no symbols are invalidated")
def nothing_invalidated(ctx: SimpleNamespace) -> None:
    assert ctx.refresh.invalidated == frozenset()


@then(parsers.parse('cached facts for "{sid}" are still valid'))
def still_valid(ctx: SimpleNamespace, sid: str) -> None:
    assert sid not in ctx.refresh.invalidated
    assert ctx.cache.is_valid(sid, ctx.ir)
    assert ctx.cache.facts(sid) == tuple(ctx.ir.facts_for(sid))


@then(parsers.parse('the cache holds no facts for "{sid}"'))
def no_facts(ctx: SimpleNamespace, sid: str) -> None:
    assert ctx.cache.facts(sid) is None

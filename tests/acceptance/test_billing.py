from __future__ import annotations

import json
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.billing import PriceTable, UsageRecord, billing_row

scenarios("billing.feature")


@given("the default price table")
def price_table(ctx: SimpleNamespace) -> None:
    ctx.table = PriceTable.load()


@given(
    parsers.parse(
        "{model} usage with {input} input, {cached} cached input, "
        "{write} cache write, {output} output and {reasoning} reasoning tokens"
    )
)
def usage_record(
    ctx: SimpleNamespace, model: str, input: str, cached: str, write: str, output: str, reasoning: str
) -> None:
    ctx.model = model
    ctx.usage = UsageRecord(
        model=model,
        input_tokens=int(input),
        cached_input_tokens=int(cached),
        cache_write_tokens=int(write),
        output_tokens=int(output),
        reasoning_tokens=int(reasoning),
    )


@when("the cost is computed")
def compute_cost(ctx: SimpleNamespace) -> None:
    ctx.breakdown = ctx.table.cost(ctx.model, ctx.usage)


@given("its cost is computed")
def compute_cost_given(ctx: SimpleNamespace) -> None:
    compute_cost(ctx)


@when(parsers.parse("the cost is computed for unknown model {model}"))
def compute_cost_unknown_model(ctx: SimpleNamespace, model: str) -> None:
    try:
        ctx.table.cost(model, ctx.usage)
    except KeyError as error:
        ctx.error = error
    else:
        ctx.error = None


@when("the same token volume is billed with no cache hits")
def compute_uncached_cost(ctx: SimpleNamespace) -> None:
    ctx.cached_total = ctx.breakdown.total
    uncached = UsageRecord(
        model=ctx.model,
        input_tokens=ctx.usage.input_tokens + ctx.usage.cached_input_tokens,
        output_tokens=ctx.usage.output_tokens,
    )
    ctx.uncached_breakdown = ctx.table.cost(ctx.model, uncached)
    ctx.breakdown = ctx.uncached_breakdown


@when(parsers.parse("a usage record is built with {tokens} input tokens"))
def build_negative_usage(ctx: SimpleNamespace, tokens: str) -> None:
    try:
        UsageRecord(model="claude-sonnet-4-5", input_tokens=int(tokens), output_tokens=0)
    except ValueError as error:
        ctx.error = error
    else:
        ctx.error = None


@when(parsers.parse('a billing row is emitted for run "{run_id}", task "{task_id}", turn {turn:d}'))
def emit_billing_row(ctx: SimpleNamespace, run_id: str, task_id: str, turn: int) -> None:
    ctx.row = billing_row(run_id, task_id, turn, ctx.usage, ctx.breakdown)


@when("the row is serialized and parsed back through JSON")
def row_round_trip(ctx: SimpleNamespace) -> None:
    ctx.row = json.loads(json.dumps(ctx.row))


@then(parsers.parse('the itemized "{category}" cost is {amount:f}'))
def itemized_cost(ctx: SimpleNamespace, category: str, amount: float) -> None:
    assert round(ctx.breakdown.items[category], 6) == round(amount, 6)


@then(parsers.parse("the total cost is {amount:f}"))
def total_cost(ctx: SimpleNamespace, amount: float) -> None:
    assert round(ctx.breakdown.total, 6) == round(amount, 6)


@then("the cached turn is cheaper than the uncached turn")
def cached_cheaper(ctx: SimpleNamespace) -> None:
    assert ctx.cached_total < ctx.uncached_breakdown.total


@then(parsers.parse('a KeyError names the unknown model and lists "{first}" and "{second}"'))
def unknown_model_error(ctx: SimpleNamespace, first: str, second: str) -> None:
    assert isinstance(ctx.error, KeyError)
    message = ctx.error.args[0]
    assert "gpt-4o-mini" in message
    assert first in message
    assert second in message


@then("a ValueError names the offending field")
def negative_usage_error(ctx: SimpleNamespace) -> None:
    assert isinstance(ctx.error, ValueError)
    assert "input_tokens" in ctx.error.args[0]


@then("the row carries the tokens, the itemized costs and the total")
def row_content(ctx: SimpleNamespace) -> None:
    row = ctx.row
    assert row["run_id"] == "r1"
    assert row["task_id"] == "t1"
    assert row["turn"] == 3
    assert row["model"] == "claude-sonnet-4-5"
    assert row["input_tokens"] == 10000
    assert row["cached_input_tokens"] == 90000
    assert row["cache_write_tokens"] == 0
    assert row["output_tokens"] == 1000
    assert row["reasoning_tokens"] == 0
    assert round(row["cost_input_usd"], 6) == 0.03
    assert round(row["cost_cached_input_usd"], 6) == 0.027
    assert round(row["cost_total_usd"], 6) == round(ctx.breakdown.total, 6)

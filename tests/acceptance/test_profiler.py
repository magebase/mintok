from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.profiler import FORMULAS, profile_sessions
from mintok.records import ContextRead, SessionRecord, dump_sessions_jsonl
from tests.acceptance.helpers import table_rows

scenarios("profiler.feature")


def _sample_sessions() -> list[SessionRecord]:
    return [
        SessionRecord(
            session_id="s1",
            task_id="t1",
            task_class="schema_change",
            model="opus-frontier",
            input_tokens=36000,
            output_tokens=4000,
            cache_read_tokens=30000,
            usd=40.00,
            context_reads=(ContextRead(key="billing.py", tokens=4000, count=4),),
        ),
        SessionRecord(
            session_id="s2",
            task_id="t2",
            task_class="deterministic",
            model="opus-frontier",
            input_tokens=20000,
            output_tokens=5000,
            cache_read_tokens=20000,
            usd=30.00,
            context_reads=(ContextRead(key="api.py", tokens=1000, count=1),),
        ),
        SessionRecord(
            session_id="s3",
            task_id="t3",
            task_class="bugfix",
            model="opus-frontier",
            input_tokens=10000,
            output_tokens=4200,
            cache_read_tokens=1000,
            usd=14.20,
            context_reads=(ContextRead(key="util.py", tokens=2000, count=2),),
        ),
    ]


@given("these agent session records:")
def session_records(ctx: SimpleNamespace, datatable: list[list[str]]) -> None:
    sessions = []
    for row in table_rows(datatable):
        reads = []
        for cell in row["context_reads"].split(", "):
            key, tokens, count = cell.rsplit(":", 2)
            reads.append(ContextRead(key=key, tokens=int(tokens), count=int(count)))
        sessions.append(
            SessionRecord(
                session_id=row["session"],
                task_id=row["task"],
                task_class=row["class"],
                model=row["model"],
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cache_read_tokens=int(row["cache_read_tokens"]),
                usd=float(row["usd"]),
                context_reads=tuple(reads),
            )
        )
    ctx.sessions = sessions


@when("the profile is computed")
def compute_profile(ctx: SimpleNamespace) -> None:
    ctx.profile = profile_sessions(ctx.sessions)


@then(parsers.parse('total spend is "${total:g}"'))
def total_spend(ctx: SimpleNamespace, total: float) -> None:
    assert round(ctx.profile.total, 2) == total


@then(parsers.parse('avoidable "{category}" is "${amount:g}"'))
def category_amount(ctx: SimpleNamespace, category: str, amount: float) -> None:
    assert round(ctx.profile.categories[category], 2) == amount


@then(parsers.parse('estimated avoidable is "${amount:g}"'))
def estimated_avoidable(ctx: SimpleNamespace, amount: float) -> None:
    assert round(ctx.profile.avoidable, 2) == amount


@then(parsers.parse('the potential improvement is "{multiple:g}x"'))
def improvement(ctx: SimpleNamespace, multiple: float) -> None:
    assert round(ctx.profile.improvement_multiple, 2) == multiple


@then("the potential improvement is unbounded")
def improvement_unbounded(ctx: SimpleNamespace) -> None:
    import math

    assert math.isinf(ctx.profile.improvement_multiple)


@then("\"repeated context\" prices every read beyond the first at the session's average cost per token")
def formula_repeated_context() -> None:
    assert (
        FORMULAS["repeated context"]
        == "price every read beyond the first at the session's average cost per token"
    )


@then('"overpowered model" flags the full cost of frontier sessions on deterministic task classes')
def formula_overpowered() -> None:
    assert (
        FORMULAS["overpowered model"]
        == "flags the full cost of frontier sessions on deterministic task classes"
    )


@then('"poor cache use" flags 25% of cost when fewer than half the input tokens hit cache')
def formula_cache() -> None:
    assert FORMULAS["poor cache use"] == "flags 25% of cost when fewer than half the input tokens hit cache"


@given("a session JSONL file with $84.20 total spend")
def session_file(ctx: SimpleNamespace, repo: Path) -> None:
    dump_sessions_jsonl(_sample_sessions(), repo / "sessions.jsonl")

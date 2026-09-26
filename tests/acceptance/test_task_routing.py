from __future__ import annotations

import re
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.router import PreFlightFeatures, prediction_record, route

scenarios("task_routing.feature")


@given(parsers.parse('an instruction "{text}"'))
def instruction(ctx: SimpleNamespace, text: str) -> None:
    ctx.instruction = text
    ctx.sizes = {}


@given(parsers.parse('target files: {specs}'))
def target_files(ctx: SimpleNamespace, specs: str) -> None:
    ctx.sizes = {path: int(loc) for path, loc in re.findall(r'"([^"]+)"=(\d+)', specs)}


@when("the task is routed")
def route_task(ctx: SimpleNamespace) -> None:
    ctx.decision = route(PreFlightFeatures(ctx.instruction, ctx.sizes))


@when(parsers.parse(
    'the task "{task_id}" is routed with actual outcomes '
    "control={c_tok:d}/{c_ok} and semantic-C={s_tok:d}/{s_ok}"
))
def route_with_actual(ctx: SimpleNamespace, task_id: str, c_tok: int, c_ok: str, s_tok: int, s_ok: str) -> None:
    ctx.decision = route(PreFlightFeatures(ctx.instruction, ctx.sizes))
    ctx.record = prediction_record(
        task_id,
        PreFlightFeatures(ctx.instruction, ctx.sizes),
        ctx.decision,
        actual={
            "control": {"tokens": c_tok, "solved": c_ok == "solved"},
            "semantic-C": {"tokens": s_tok, "solved": s_ok == "solved"},
        },
    )


@then(parsers.parse('the backend is "{backend}"'))
def backend_is(ctx: SimpleNamespace, backend: str) -> None:
    assert ctx.decision.backend == backend, ctx.decision


@then(parsers.parse('the predicted class is "{klass}"'))
def predicted_class_is(ctx: SimpleNamespace, klass: str) -> None:
    assert ctx.decision.predicted_class == klass, ctx.decision


@then(parsers.parse("the expected relative cost is below {bound:f}"))
def cost_below(ctx: SimpleNamespace, bound: float) -> None:
    assert ctx.decision.expected_relative_cost < bound, ctx.decision


@then(parsers.parse('the reasons include "{text}"'))
def reasons_include(ctx: SimpleNamespace, text: str) -> None:
    assert any(text in reason for reason in ctx.decision.reasons), ctx.decision.reasons


@then(parsers.parse("the confidence is below {bound:f}"))
def confidence_below(ctx: SimpleNamespace, bound: float) -> None:
    assert ctx.decision.confidence < bound, ctx.decision


@then(parsers.parse('the record selects "{backend}"'))
def record_selects(ctx: SimpleNamespace, backend: str) -> None:
    assert ctx.record["backend"] == backend, ctx.record


@then(parsers.parse('the record contains the feature "target_loc" with value {value:d}'))
def record_has_target_loc(ctx: SimpleNamespace, value: int) -> None:
    assert ctx.record["features"]["target_loc"] == value, ctx.record


@then(parsers.parse('the record contains "actual" for both backends'))
def record_has_actual(ctx: SimpleNamespace) -> None:
    assert set(ctx.record["actual"]) == {"control", "semantic-C"}, ctx.record

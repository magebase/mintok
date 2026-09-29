from __future__ import annotations

import re
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.repo_profile import RepoProfile
from mintok.router import PreFlightFeatures, compute_utility, prediction_record, route

scenarios("task_routing.feature")


@given(parsers.parse('an instruction "{text}"'))
def instruction(ctx: SimpleNamespace, text: str) -> None:
    ctx.instruction = text
    ctx.sizes = {}
    ctx.repo_profile = None


@given(parsers.parse('target files: {specs}'))
def target_files(ctx: SimpleNamespace, specs: str) -> None:
    ctx.sizes = {path: int(loc) for path, loc in re.findall(r'"([^"]+)"=(\d+)', specs)}


@given(parsers.parse('a repository profile with complexity {complexity:f} and strategy "{strategy}"'))
def repo_profile_given(ctx: SimpleNamespace, complexity: float, strategy: str) -> None:
    ctx.repo_profile = RepoProfile(
        repo_name="test-repo",
        complexity_score=complexity,
        recommended_strategy=strategy,
    )


@when("the task is routed")
def route_task(ctx: SimpleNamespace) -> None:
    ctx.decision = route(PreFlightFeatures(ctx.instruction, ctx.sizes, getattr(ctx, "repo_profile", None)))


@when(parsers.parse(
    'the task "{task_id}" is routed with actual outcomes '
    "control={c_tok:d}/{c_ok} and semantic-C={s_tok:d}/{s_ok}"
))
def route_with_actual(ctx: SimpleNamespace, task_id: str, c_tok: int, c_ok: str, s_tok: int, s_ok: str) -> None:
    feats = PreFlightFeatures(ctx.instruction, ctx.sizes, getattr(ctx, "repo_profile", None))
    ctx.decision = route(feats)
    ctx.record = prediction_record(
        task_id,
        feats,
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


@then(parsers.parse('the utility for "{backend1}" is higher than "{backend2}"'))
def utility_higher(ctx: SimpleNamespace, backend1: str, backend2: str) -> None:
    assert ctx.record["utilities"][backend1] > ctx.record["utilities"][backend2], ctx.record["utilities"]


@then(parsers.re(r"the expected utility is greater than (?P<val>[\d.]+)"))
def expected_utility_greater(ctx: SimpleNamespace, val: str):
    assert ctx.decision.expected_utility > float(val)


@then(parsers.re(r'the decision includes policy expectation for "(?P<policy>[^"]+)" with success probability above (?P<prob>[\d.]+)'))
def policy_expectation_check(ctx: SimpleNamespace, policy: str, prob: str):
    assert policy in ctx.decision.policy_expectations
    exp = ctx.decision.policy_expectations[policy]
    assert exp.p_success > float(prob)



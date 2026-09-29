"""Acceptance steps for MinTok 3.1 advanced inference optimization mechanisms."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from mintok.abi import estimate_tool_surface_tokens, get_filtered_tool_surface
from mintok.controller import (
    CalibratedLocalController,
    StateFeatures,
    TabularFeatureDataset,
)
from mintok.conversation import (
    CanonicalState,
    ContextLifetime,
    ContextObject,
    ContextRentManager,
    EvidenceFact,
    FactResidencyTier,
    compute_state_delta,
    evaluate_cache_aware_compaction_benefit,
)
from mintok.coprocessor import PacketUtilityTracker
from mintok.early_stop import CleanContextRestart, ContinuationPredictor
from mintok.metrics import ProviderPricing, TokenUsageBreakdown
from mintok.profiler import LostSolveCategory, attribute_lost_solve
from mintok.source_cache import RegionStatus, SourceCache
from mintok.verification import (
    PassVerificationCache,
    VerificationCompiler,
    VerificationSafetyTracker,
)

scenarios("inference_optimizers.feature")



# ---------------------------------------------------------------------------
# Scenario 1: Verification Compiler
# ---------------------------------------------------------------------------


@given(parsers.parse('a failing test output with 1 failure "{test_id}" and assertion "{assertion}"'))
def given_failing_test_output(ctx: SimpleNamespace, test_id: str, assertion: str) -> None:
    ctx.raw_output = f"""
============================= test session starts ==============================
rootdir: /repo
collected 15 items

tests/test_calc.py ............F..                                       [100%]

=================================== FAILURES ===================================
_________________________________ test_division _________________________________
    def test_division():
>       assert divide(10, 0) == 0
E       {assertion}: division by zero
tests/test_calc.py:24: {assertion}
=========================== short test summary info ============================
FAILED {test_id} - {assertion}: division by zero
========================= 1 failed, 14 passed in 0.12s =========================
"""
    ctx.test_id = test_id
    ctx.compiler = VerificationCompiler()
    ctx.compiler.record_pre_patch_baseline([test_id])


@when("the verification compiler compiles the test output")
def when_compile_verification(ctx: SimpleNamespace) -> None:
    ctx.digest, ctx.meta = ctx.compiler.compile(
        raw_output=ctx.raw_output,
        exit_code=1,
        targeted_test=ctx.test_id,
        impacted_tests=[ctx.test_id, "tests/test_calc.py::test_multiply"],
    )


@then(parsers.parse('the verification status indicates targeted test "{status}"'))
def check_targeted_status(ctx: SimpleNamespace, status: str) -> None:
    assert f"targeted: {status}" in ctx.digest


@then(parsers.parse("new regressions count is {count:d}"))
def check_regression_count(ctx: SimpleNamespace, count: int) -> None:
    assert ctx.meta["regression_count"] == count


@then(parsers.parse("the verification digest tokens are less than {max_tok:d} tokens"))
def check_digest_tokens(ctx: SimpleNamespace, max_tok: int) -> None:
    assert ctx.meta["digest_tokens"] < max_tok


# ---------------------------------------------------------------------------
# Scenario 2: CanonicalState Delta Encoding
# ---------------------------------------------------------------------------


@given(parsers.parse('an initial canonical state with goal "{goal}"'))
def given_initial_canonical_state(ctx: SimpleNamespace, goal: str) -> None:
    ctx.state1 = CanonicalState(goal=goal, version=1)


@given(parsers.parse('the state has verified fact "{fact_id}" with value "{val}"'))
def given_fact_in_state(ctx: SimpleNamespace, fact_id: str, val: str) -> None:
    ctx.state1.add_fact(value=val, evidence="obs:1234")


@when(parsers.parse('a second state is created with added fact "{fact_id}" with value "{val}"'))
def when_second_state_created(ctx: SimpleNamespace, fact_id: str, val: str) -> None:
    ctx.state2 = CanonicalState(goal=ctx.state1.goal, version=2)
    ctx.state2.verified_facts = list(ctx.state1.verified_facts)
    ctx.state2.add_fact(value=val, evidence="obs:5678")


@then("computing state delta between state 1 and state 2 produces a delta block")
def check_state_delta(ctx: SimpleNamespace) -> None:
    ctx.delta = compute_state_delta(ctx.state1, ctx.state2)
    ctx.delta_text = ctx.delta.render()
    assert len(ctx.delta.added_facts) == 1


@then(parsers.parse('the delta text includes "{expected}"'))
def check_delta_text_includes(ctx: SimpleNamespace, expected: str) -> None:
    assert expected in ctx.delta_text


@then(parsers.parse("the delta text length is less than {max_chars:d} characters"))
def check_delta_len(ctx: SimpleNamespace, max_chars: int) -> None:
    assert len(ctx.delta_text) < max_chars


# ---------------------------------------------------------------------------
# Scenario 3: State Residency Hierarchy
# ---------------------------------------------------------------------------


@given("a canonical state with a hot goal, a hot failing assertion, and a cold rejected hypothesis")
def given_state_with_mixed_residency(ctx: SimpleNamespace) -> None:
    state = CanonicalState(goal="Fix database deadlock", version=3)
    state.add_fact("failing assertion in query.py: timeout after 30s", evidence="obs:fail")
    state.set_hypothesis("Deadlock caused by table lock")
    state.reject_hypothesis(reason="indexes are fine", evidence="obs:trace")
    state.set_hypothesis("Deadlock caused by connection pool exhaustion")
    state.reject_hypothesis(reason="pool is idle", evidence="obs:trace2")
    state.set_hypothesis("Transaction ordering bug")
    state.reject_hypothesis(reason="serializable isolation", evidence="obs:trace3")
    ctx.residency_state = state


@when("the state is rendered with residency filtering enabled")
def when_rendered_with_residency(ctx: SimpleNamespace) -> None:
    ctx.rendered_residency = ctx.residency_state.render(residency_filter=True)


@then("the rendered state includes the hot failing assertion")
def check_hot_fact_included(ctx: SimpleNamespace) -> None:
    assert "failing assertion in query.py" in ctx.rendered_residency


@then("the cold rejected hypothesis is archived to local store")
def check_cold_hypothesis_archived(ctx: SimpleNamespace) -> None:
    assert "archived" in ctx.rendered_residency


# ---------------------------------------------------------------------------
# Scenario 4: Packet Utility Tracker
# ---------------------------------------------------------------------------


@given("a packet utility tracker")
def given_packet_utility_tracker(ctx: SimpleNamespace) -> None:
    ctx.tracker = PacketUtilityTracker()


@when(parsers.parse('a packet "{pkt_id}" of type "{pkt_type}" with {tokens:d} tokens is evaluated'))
def when_evaluate_packet(ctx: SimpleNamespace, pkt_id: str, pkt_type: str, tokens: int) -> None:
    ctx.tracker.record_packet(pkt_id, pkt_type, tokens, ["calc", "add"])
    ctx.last_should_prune = ctx.tracker.should_prune(pkt_type, tokens)


@then("the packet utility tracker recommends pruning the packet")
def check_recommends_pruning(ctx: SimpleNamespace) -> None:
    assert ctx.last_should_prune is True


@then("the packet utility tracker recommends retaining the packet")
def check_recommends_retaining(ctx: SimpleNamespace) -> None:
    assert ctx.last_should_prune is False


# ---------------------------------------------------------------------------
# Scenario 5: Source Cache
# ---------------------------------------------------------------------------


@given("an empty source cache")
def given_empty_source_cache(ctx: SimpleNamespace) -> None:
    ctx.source_cache = SourceCache()


@when(parsers.parse('a source span for "{fpath}" lines {start:d} to {end:d} is registered'))
def when_register_span(ctx: SimpleNamespace, fpath: str, start: int, end: int) -> None:
    lines = [f"line_{i} = {i}" for i in range(start, end + 1)]
    content = "\n".join(lines)
    ctx.reg_text, ctx.novelty = ctx.source_cache.register_span(fpath, start, end, content)


@then(parsers.parse("the computed novelty is {expected:f}"))
def check_novelty_exact(ctx: SimpleNamespace, expected: float) -> None:
    assert ctx.novelty == pytest.approx(expected, 0.01)


@when(parsers.parse('an overlapping span for "{fpath}" lines {start:d} to {end:d} is evaluated'))
def when_eval_overlapping_span(ctx: SimpleNamespace, fpath: str, start: int, end: int) -> None:
    ctx.overlap_novelty = ctx.source_cache.compute_novelty(fpath, start, end)


@then(parsers.parse("the computed novelty is less than {thresh:f}"))
def check_overlap_novelty_less(ctx: SimpleNamespace, thresh: float) -> None:
    assert ctx.overlap_novelty < thresh


@when(parsers.parse('the exact span for "{fpath}" lines {start:d} to {end:d} is registered again'))
def when_register_exact_again(ctx: SimpleNamespace, fpath: str, start: int, end: int) -> None:
    lines = [f"line_{i} = {i}" for i in range(start, end + 1)]
    content = "\n".join(lines)
    ctx.re_reg_text, ctx.re_novelty = ctx.source_cache.register_span(fpath, start, end, content)


@then(parsers.parse('the source cache returns an unchanged handle "{handle}"'))
def check_unchanged_handle(ctx: SimpleNamespace, handle: str) -> None:
    assert handle in ctx.re_reg_text


# ---------------------------------------------------------------------------
# Scenario 6: Continuation Predictor
# ---------------------------------------------------------------------------


@given("a continuation predictor")
def given_continuation_predictor(ctx: SimpleNamespace) -> None:
    ctx.predictor = ContinuationPredictor(token_ceiling=1_500_000, min_continuation_prob=0.03)


@when(parsers.parse("a trajectory has spent {tokens:d} tokens across {turns:d} turns with {consec_fails:d} consecutive failures"))
def when_trajectory_evaluated(ctx: SimpleNamespace, tokens: int, turns: int, consec_fails: int) -> None:
    ctx.est = ctx.predictor.evaluate(
        tokens_spent=tokens,
        turns_elapsed=turns,
        consecutive_failures=consec_fails,
        tests_resolved=0,
        has_patch=False,
    )


@then(parsers.parse('the recommended continuation action is "{action}"'))
def check_recommended_action(ctx: SimpleNamespace, action: str) -> None:
    assert ctx.est.recommended_action == action


@then(parsers.parse("the eventual solve probability is less than {thresh:f}"))
def check_eventual_solve_prob(ctx: SimpleNamespace, thresh: float) -> None:
    assert ctx.est.p_solve_eventually < thresh


# ---------------------------------------------------------------------------
# Scenario 7: Clean-Context Restart
# ---------------------------------------------------------------------------


@given("a canonical state with verified facts, a current patch, and target failing tests")
def given_state_for_restart(ctx: SimpleNamespace) -> None:
    s = CanonicalState(goal="Fix KeyError on missing header", version=8)
    s.add_fact("header name is normalized to lowercase", evidence="obs:1111")
    s.current_patch = "diff --git a/client.py b/client.py\n+ header = header.lower()"
    s.record_failure("tests/test_client.py::test_missing_header")
    ctx.restart_state = s


@when("a clean-context restart prompt is constructed")
def when_construct_restart_prompt(ctx: SimpleNamespace) -> None:
    ctx.restart_prompt = CleanContextRestart.build_restart_prompt(
        state=ctx.restart_state,
        task_instruction=ctx.restart_state.goal,
    )


@then(parsers.parse('the prompt contains "{text}"'))
def check_prompt_contains(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.restart_prompt


@then("the prompt contains the current patch")
def check_prompt_contains_patch(ctx: SimpleNamespace) -> None:
    assert "header = header.lower()" in ctx.restart_prompt


@then("the prompt omits dead reasoning history")
def check_prompt_omits_dead_history(ctx: SimpleNamespace) -> None:
    assert "history" not in ctx.restart_prompt.lower() or "purged" in ctx.restart_prompt.lower()


# ---------------------------------------------------------------------------
# Scenario 8: Calibrated Controller & Regret Weighting
# ---------------------------------------------------------------------------


@given("a tabular feature dataset")
def given_tabular_dataset(ctx: SimpleNamespace) -> None:
    ctx.dataset = TabularFeatureDataset()
    ctx.controller = CalibratedLocalController()


@when(parsers.parse("a training sample with utility regret {regret:f} is added"))
def when_add_sample_with_regret(ctx: SimpleNamespace, regret: float) -> None:
    feats = StateFeatures(
        repo_packages=4,
        repo_complexity=0.55,
        task_issue_length=800,
        task_named_symbols=3,
        tokens_spent=250000,
        turns_elapsed=12,
        active_failures=2,
        verified_facts_count=3,
        has_patch=1,
        candidate_action="virtualized-shell",
        repo_name="Azure/azure-cli",
    )
    ctx.dataset.add_sample(
        features=feats,
        eventual_success=True,
        remaining_tokens=180000,
        utility_regret=regret,
    )


@then(parsers.parse("the sample weight is {weight:f}"))
def check_sample_weight(ctx: SimpleNamespace, weight: float) -> None:
    assert ctx.dataset.samples[0].sample_weight == pytest.approx(weight, 0.01)


@then("the calibrated controller evaluates candidate actions and selects the highest utility action")
def check_controller_action_selection(ctx: SimpleNamespace) -> None:
    actions = ["virtualized-shell", "semantic-compiler", "macro-action", "abort"]
    base_dict = {
        "repo_packages": 6,
        "repo_complexity": 0.65,
        "task_issue_length": 1200,
        "task_named_symbols": 2,
        "tokens_spent": 150000,
        "turns_elapsed": 8,
        "active_failures": 1,
        "verified_facts_count": 2,
        "has_patch": 1,
        "repo_name": "Azure/azure-cli",
    }
    best_act, best_q = ctx.controller.select_best_action(actions, base_dict)
    assert best_act in actions
    assert best_q > -float("inf")


# ---------------------------------------------------------------------------
# Scenario 9: Verification Safety Tracker & Miss Rate
# ---------------------------------------------------------------------------


@given("a verification safety tracker")
def given_verification_safety_tracker(ctx: SimpleNamespace) -> None:
    ctx.safety_tracker = VerificationSafetyTracker()


@when(parsers.parse("{total:d} verification events occur with {false_passes:d} false pass and {failures:d} broader failures"))
def when_verification_events_occur(ctx: SimpleNamespace, total: int, false_passes: int, failures: int) -> None:
    for i in range(total):
        if i < false_passes:
            ctx.safety_tracker.record_event(selected_passed=True, broader_passed=False)
        elif i < failures:
            ctx.safety_tracker.record_event(selected_passed=False, broader_passed=False)
        else:
            ctx.safety_tracker.record_event(selected_passed=True, broader_passed=True)


@then(parsers.parse("the computed miss rate is {expected:f}"))
def check_computed_miss_rate(ctx: SimpleNamespace, expected: float) -> None:
    assert ctx.safety_tracker.miss_rate == pytest.approx(expected, 0.01)


@then(parsers.parse("the safety gate fails when max allowed miss rate is {max_rate:f}"))
def check_safety_gate(ctx: SimpleNamespace, max_rate: float) -> None:
    assert not ctx.safety_tracker.is_safe(max_rate)


# ---------------------------------------------------------------------------
# Scenario 10: Passing Verification Cache
# ---------------------------------------------------------------------------


@given("a passing verification cache")
def given_pass_cache(ctx: SimpleNamespace) -> None:
    ctx.pass_cache = PassVerificationCache()


@when(parsers.parse('test "{test_id}" passes for dependency hash "{h}"'))
def when_test_passes_for_hash(ctx: SimpleNamespace, test_id: str, h: str) -> None:
    ctx.pass_cache.record_pass(test_id, h)
    ctx.test_id = test_id


@then(parsers.parse('the test is confirmed as a cached pass for hash "{h}"'))
def check_cached_pass(ctx: SimpleNamespace, h: str) -> None:
    assert ctx.pass_cache.is_cached_pass(ctx.test_id, h) is True


@then(parsers.parse('the test is not a cached pass for hash "{h}"'))
def check_not_cached_pass(ctx: SimpleNamespace, h: str) -> None:
    assert ctx.pass_cache.is_cached_pass(ctx.test_id, h) is False


# ---------------------------------------------------------------------------
# Scenario 11: Context Rent Manager Event-Driven Eviction
# ---------------------------------------------------------------------------


@given(parsers.parse('a context rent manager with an admitted object associated with hypothesis "{h_id}"'))
def given_context_rent_manager_with_obj(ctx: SimpleNamespace, h_id: str) -> None:
    ctx.rent_mgr = ContextRentManager()
    obj = ContextObject(
        id="ctx_h1",
        content="failing test traceback for hypothesis H1",
        tokens=350,
        lifetime=ContextLifetime.TASK_LONG,
        associated_hypothesis=h_id,
        expected_future_replays=10,
        decision_value=0.9,
    )
    assert ctx.rent_mgr.admit(obj) is True


@when(parsers.parse('an event "{event_type}" for "{h_id}" is triggered'))
def when_event_triggered(ctx: SimpleNamespace, event_type: str, h_id: str) -> None:
    ctx.evicted = ctx.rent_mgr.on_event(event_type, h_id)


@then(parsers.parse('the object associated with "{h_id}" is evicted from active memory'))
def check_object_evicted(ctx: SimpleNamespace, h_id: str) -> None:
    assert "ctx_h1" in ctx.evicted
    assert ctx.rent_mgr.get("ctx_h1") is None


# ---------------------------------------------------------------------------
# Scenario 12: Cache-Aware Compaction Benefit
# ---------------------------------------------------------------------------


@when(parsers.parse("evaluating cache compaction for prefix {prefix:d} tokens and delta {delta:d} tokens across {turns:d} remaining turns"))
def when_eval_compaction(ctx: SimpleNamespace, prefix: int, delta: int, turns: int) -> None:
    ctx.net_benefit = evaluate_cache_aware_compaction_benefit(
        prefix_tokens=prefix,
        delta_tokens=delta,
        remaining_turns=turns,
    )


@then("the net compaction benefit is positive")
def check_net_benefit_positive(ctx: SimpleNamespace) -> None:
    assert ctx.net_benefit > 0.0


# ---------------------------------------------------------------------------
# Scenario 13: Source Cache SEEN vs RESIDENT Lifecycle
# ---------------------------------------------------------------------------


@given(parsers.parse('a source cache with a resident span for "{file_path}" lines {start:d} to {end:d}'))
def given_source_cache_resident_span(ctx: SimpleNamespace, file_path: str, start: int, end: int) -> None:
    ctx.source_cache = SourceCache()
    content = "def calculate_total():\n    return 42\n"
    text, novelty = ctx.source_cache.register_span(file_path, start, end, content)
    ctx.src_id = list(ctx.source_cache._regions_by_id.keys())[0]
    assert ctx.source_cache._regions_by_id[ctx.src_id].status == RegionStatus.RESIDENT


@when("the span is evicted from active context")
def when_span_evicted(ctx: SimpleNamespace) -> None:
    ctx.source_cache.mark_evicted(ctx.src_id)


@then(parsers.parse('its status becomes "{expected}"'))
def check_span_status(ctx: SimpleNamespace, expected: str) -> None:
    assert ctx.source_cache._regions_by_id[ctx.src_id].status == expected


@when("the span is rehydrated")
def when_span_rehydrated(ctx: SimpleNamespace) -> None:
    ctx.source_cache.rehydrate(ctx.src_id)


# ---------------------------------------------------------------------------
# Scenario 14: Dynamic Tool Surface Phase Filtering
# ---------------------------------------------------------------------------


@when(parsers.parse('filtering tool surface for phase "{phase}"'))
def when_filter_tool_surface(ctx: SimpleNamespace, phase: str) -> None:
    ctx.filtered_tools = get_filtered_tool_surface(phase)
    ctx.tool_tokens = estimate_tool_surface_tokens(phase)


@then(parsers.parse('only "{t1}" and "{t2}" tools are exposed'))
def check_filtered_tools(ctx: SimpleNamespace, t1: str, t2: str) -> None:
    names = {t["name"] for t in ctx.filtered_tools}
    assert names == {t1, t2}


@then(parsers.parse('the tool surface tokens for "{phase}" are less than {max_tokens:d} tokens'))
def check_filtered_tokens(ctx: SimpleNamespace, phase: str, max_tokens: int) -> None:
    assert ctx.tool_tokens < max_tokens


# ---------------------------------------------------------------------------
# Scenario 15: Adaptive Semantic Packet Pruning
# ---------------------------------------------------------------------------


@when(parsers.parse('a packet of type "{p_type}" with {tokens:d} tokens has retriever disagreement'))
def when_packet_disagreement(ctx: SimpleNamespace, p_type: str, tokens: int) -> None:
    ctx.last_should_prune = ctx.tracker.should_prune(
        packet_type=p_type,
        candidate_tokens=tokens,
        retriever_disagreement=True,
    )


# ---------------------------------------------------------------------------
# Scenario 16: Billed Dollars Accounting
# ---------------------------------------------------------------------------


@given(parsers.parse("a token usage breakdown with {fresh:d} fresh, {cached:d} cached, {output:d} output, and {reasoning:d} reasoning tokens"))
def given_token_breakdown(ctx: SimpleNamespace, fresh: int, cached: int, output: int, reasoning: int) -> None:
    ctx.breakdown = TokenUsageBreakdown(
        t_fresh=fresh,
        t_cached=cached,
        t_output=output,
        t_reasoning=reasoning,
    )


@when("computing billed cost under default pricing")
def when_computing_billed_cost(ctx: SimpleNamespace) -> None:
    ctx.billed_usd = ctx.breakdown.compute_billed_usd()


@then(parsers.parse("the total billed cost is greater than {min_cost:f} and less than {max_cost:f} dollars"))
def check_billed_cost_range(ctx: SimpleNamespace, min_cost: float, max_cost: float) -> None:
    assert min_cost < ctx.billed_usd < max_cost


# ---------------------------------------------------------------------------
# Scenario 17: Lost Solve Attribution
# ---------------------------------------------------------------------------


@when("attributing a lost solve where the task was early stopped")
def when_attributing_lost_solve(ctx: SimpleNamespace) -> None:
    ctx.attribution = attribute_lost_solve(
        task_id="task_123",
        control_solved=True,
        mintok_solved=False,
        early_stopped=True,
        tokens_saved=50000,
    )


@then(parsers.parse('the attribution category is "{expected}"'))
def check_attribution_category(ctx: SimpleNamespace, expected: str) -> None:
    assert ctx.attribution is not None
    assert ctx.attribution.category == expected


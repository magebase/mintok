"""Acceptance tests for Learned MinTok Controller and PolicyBench Tournament."""

from __future__ import annotations

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.learned_policy import (
    AnswerLocalizationModel,
    BypassClassifier,
    ContinuousBudgetModel,
    DestructiveActionModel,
    FirstAction,
    FirstActionClassifier,
    LearnedActionSelector,
    LearnedMinTokController,
    ModelCapabilityAdapter,
    PatchStopModel,
    PolicyState,
)
from mintok.policybench_suite import PolicyArm, PolicyBenchEvaluator, PolicyBenchTournamentReport

scenarios("learned_controller.feature")


# --- Learned Action Selector Steps ---

@given(parsers.parse('a policy state with candidate actions "{a1}", "{a2}", and "{a3}"'), target_fixture="sel_ctx")
def given_policy_state_and_candidates(a1: str, a2: str, a3: str):
    state = PolicyState(
        task_loc=120,
        target_loc=40,
        repo_complexity=0.5,
        turn=2,
        tokens_spent=3500,
        patch_lines=0,
    )
    selector = LearnedActionSelector()
    return {"state": state, "candidates": [a1, a2, a3], "selector": selector}


@when("the learned action selector evaluates marginal ROI")
def when_eval_marginal_roi(sel_ctx):
    evals = sel_ctx["selector"].evaluate_actions(sel_ctx["state"], sel_ctx["candidates"])
    best = sel_ctx["selector"].select_best_action(sel_ctx["state"], sel_ctx["candidates"])
    sel_ctx["evaluations"] = evals
    sel_ctx["best"] = best


@then(parsers.parse('"{expected_action}" achieves the highest marginal ROI ratio'))
def then_action_highest_roi(sel_ctx, expected_action: str):
    best = sel_ctx["best"]
    assert best.action == expected_action
    assert best.marginal_roi > 0.0


@then("the action maximizing marginal ROI is selected")
def then_max_roi_selected(sel_ctx):
    best = sel_ctx["best"]
    evals = sel_ctx["evaluations"]
    max_roi = max(e.marginal_roi for e in evals)
    assert best.marginal_roi == max_roi


# --- Bypass Classifier Steps ---

@given("a trivial rename task with 15 LOC and low complexity", target_fixture="bypass_ctx")
def given_trivial_task():
    classifier = BypassClassifier(controller_overhead_tokens=250)
    state = PolicyState(task_loc=15, repo_complexity=0.1)
    return {"classifier": classifier, "state": state}


@when("the bypass classifier evaluates the task")
def when_bypass_evaluates_task(bypass_ctx):
    res = bypass_ctx["classifier"].evaluate(bypass_ctx["state"], task_description="rename function foo to bar")
    bypass_ctx["result"] = res


@then("the verdict is to bypass MinTok because expected savings are less than controller overhead")
def then_bypass_verdict(bypass_ctx):
    res = bypass_ctx["result"]
    assert res.should_bypass is True
    assert res.expected_savings_tokens <= res.controller_overhead_tokens


@when("a complex multi-file task with 250 LOC is evaluated")
def when_complex_task_evaluated(bypass_ctx):
    complex_state = PolicyState(task_loc=250, repo_complexity=0.7)
    res = bypass_ctx["classifier"].evaluate(complex_state, task_description="refactor payment processing pipeline")
    bypass_ctx["complex_result"] = res


@then("the bypass classifier determines MinTok should be engaged")
def then_bypass_engaged(bypass_ctx):
    res = bypass_ctx["complex_result"]
    assert res.should_bypass is False
    assert res.net_savings_tokens > 0


# --- Task Topology First Action Classifier Steps ---

@given("a task description with failing test traceback", target_fixture="topology_ctx")
def given_failing_test_task():
    return {"task": "fix AssertionError in tests/test_billing.py: line 45 assert 100 == 120"}


@when("the first action classifier evaluates the task")
def when_first_action_evaluates(topology_ctx):
    act, reason = FirstActionClassifier.classify(topology_ctx["task"], {})
    topology_ctx["action"] = act
    topology_ctx["reason"] = reason


@then(parsers.parse('the recommended first action is "{expected_act}"'))
def then_first_action_matches(topology_ctx, expected_act: str):
    assert topology_ctx["action"].value == expected_act


@when("a task description specifies an API signature regression")
def when_api_task_evaluated(topology_ctx):
    topology_ctx["task"] = "update public API signature callers in auth/client.py"
    act, reason = FirstActionClassifier.classify(topology_ctx["task"], {})
    topology_ctx["action"] = act


# --- Aggressive Patch Stop Model Steps ---

@given("an applied patch with passing unit tests", target_fixture="stop_ctx")
def given_applied_patch_passing():
    state = PolicyState(patch_lines=12, tests_passing=True, hypothesis_confidence=0.85)
    model = PatchStopModel(confidence_threshold=0.80)
    return {"state": state, "model": model}


@when("the patch stop model evaluates the state")
def when_stop_model_evaluates(stop_ctx):
    verdict = stop_ctx["model"].evaluate(
        state=stop_ctx["state"],
        patch_applied=True,
        verification_passed=True,
        tests_passing=True,
    )
    stop_ctx["verdict"] = verdict


@then(parsers.parse('the directive is "{directive}" with P(correct) exceeding {p_floor:f}'))
def then_stop_directive(stop_ctx, directive: str, p_floor: float):
    v = stop_ctx["verdict"]
    assert v.should_stop is True
    assert v.action_directive == directive
    assert v.p_correct >= p_floor


# --- Destructive Action Model Steps ---

@given("a proposed patch modifying 5 files, 120 lines, and public API", target_fixture="destruct_ctx")
def given_risky_patch():
    return {
        "patch_loc": 120,
        "files_touched": 5,
        "symbols_touched": 3,
        "public_api": True,
    }


@when("the destructive action model evaluates regression risk")
def when_eval_destructive_risk(destruct_ctx):
    model = DestructiveActionModel(risk_threshold=0.40)
    report = model.evaluate_risk(
        patch_loc=destruct_ctx["patch_loc"],
        files_touched=destruct_ctx["files_touched"],
        symbols_touched=destruct_ctx["symbols_touched"],
        public_api_touched=destruct_ctx["public_api"],
        unrelated_lines_touched=15,
        distance_from_hypothesis=0.6,
        previous_patches_count=2,
        failed_verification_count=1,
        patch_entropy=0.8,
    )
    destruct_ctx["report"] = report


@then(parsers.parse('the regression risk exceeds {threshold:f} and review is enforced'))
def then_regression_risk_high(destruct_ctx, threshold: float):
    rep = destruct_ctx["report"]
    assert rep.p_regression >= threshold
    assert rep.is_high_risk is True
    assert rep.enforce_review is True
    assert len(rep.risk_factors) >= 3


# --- Continuous Token Budget Model Steps ---

@given("a simple localized task state", target_fixture="budget_state_ctx")
def given_simple_localized_state():
    return {"simple_state": PolicyState(task_loc=30, repo_complexity=0.15)}


@then(parsers.parse('continuous budget allocates {budget:d} tokens'))
def then_budget_allocates_simple(budget_state_ctx, budget: int):
    b = ContinuousBudgetModel.predict_budget(budget_state_ctx["simple_state"])
    assert b == budget


@given("a high uncertainty cross-module task state")
def given_high_uncertainty_state(budget_state_ctx):
    budget_state_ctx["uncertain_state"] = PolicyState(task_loc=200, uncertainty_total=0.85, repo_complexity=0.8)


@then(parsers.parse('continuous budget scales up to {budget:d} tokens'))
def then_budget_scales_up(budget_state_ctx, budget: int):
    b = ContinuousBudgetModel.predict_budget(budget_state_ctx["uncertain_state"])
    assert b == budget


# --- Model Capability Adapter Steps ---

@given("Model A with weak verification strength", target_fixture="model_adapter_ctx")
def given_model_a():
    base_weights = {"query_symbol": 0.3, "read_slice": 0.4, "run_verifier": 0.2, "read_full": 0.1}
    return {"model_id": "model_a", "base_weights": base_weights}


@when("the model capability adapter adjusts action weights")
def when_adapter_adjusts(model_adapter_ctx):
    adj = ModelCapabilityAdapter.adapt_action_weights(
        model_id=model_adapter_ctx["model_id"],
        base_weights=model_adapter_ctx["base_weights"],
    )
    model_adapter_ctx["adjusted_weights"] = adj


@then(parsers.parse('the weight for "{action}" is increased by at least {mult:f}x'))
def then_action_weight_increased(model_adapter_ctx, action: str, mult: float):
    orig = model_adapter_ctx["base_weights"][action]
    adj = model_adapter_ctx["adjusted_weights"][action]
    assert adj >= orig * mult


# --- Answer Localization Model Steps ---

@given("a task description and failing test output", target_fixture="loc_ctx")
def given_loc_inputs():
    return {
        "task": "Fix tax rounding calculation in billing",
        "test_output": "AssertionError: in billing/tax_calculator.py: line 42",
        "repo_files": ["billing/service.py", "billing/tax_calculator.py", "billing/models.py"],
    }


@when("answer localization predicts target locations")
def when_predict_locations(loc_ctx):
    pred = AnswerLocalizationModel.predict_location(
        task_description=loc_ctx["task"],
        failing_test_output=loc_ctx["test_output"],
        repo_files=loc_ctx["repo_files"],
    )
    loc_ctx["prediction"] = pred


@then("top-1 file matches the traceback file and top-5 accuracy estimate exceeds 90 percent")
def then_localization_accurate(loc_ctx):
    pred = loc_ctx["prediction"]
    assert pred.top_files[0] == "billing/tax_calculator.py"
    assert pred.top5_accuracy_est >= 0.90


# --- PolicyBench Controller Tournament Steps ---

@given(parsers.parse('the PolicyBench tournament evaluating {count:d} decision states across 6 policy arms'), target_fixture="tournament_ctx")
def given_tournament_config(count: int):
    return {"tasks_count": count}


@when("the benchmark tournament executes")
def when_tournament_executes(tournament_ctx):
    report = PolicyBenchEvaluator.evaluate_benchmark(tasks_count=tournament_ctx["tasks_count"])
    tournament_ctx["report"] = report


@then(parsers.parse('"{cand}" achieves lower regret than "{baseline}"'))
def then_regret_lower(tournament_ctx, cand: str, baseline: str):
    rep: PolicyBenchTournamentReport = tournament_ctx["report"]
    cand_regret = rep.arms_results[cand].regret_vs_oracle
    base_regret = rep.arms_results[baseline].regret_vs_oracle
    assert cand_regret < base_regret


@then(parsers.parse('"{policy}" is confirmed on the empirical Pareto frontier'))
def then_pareto_confirmed(tournament_ctx, policy: str):
    rep: PolicyBenchTournamentReport = tournament_ctx["report"]
    assert policy in rep.pareto_frontier_arms
    assert rep.arms_results[policy].pareto_dominant is True

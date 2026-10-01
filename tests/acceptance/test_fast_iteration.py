"""Acceptance tests for fast iteration loop, Tier 0 smoke, dev-20 mini-benchmark, replay, and policy predictor."""

from __future__ import annotations

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.bench import DEV20_TASKS, PairedSuiteResult, run_suite_benchmark
from mintok.canary import (
    CANARY_TASKS,
    MINTOK_CANARY,
    SMOKE8_TASKS,
    Tier0SmokeReport,
    run_tier0_smoke_test,
)
from mintok.predictor import AgentState, CandidateAction, PolicyPredictor
from mintok.replayer import CounterfactualReplayReport, run_counterfactual_replay

scenarios("fast_iteration.feature")


# --- Tier 0 Smoke Test Steps ---

@given('candidate policy "v3" and control policy "control"', target_fixture="ctx")
def given_tier0_policies():
    return {"candidate": "v3", "control": "control", "forced_catastrophe": False}


@given(parsers.parse('candidate policy "{cand}" with a forced catastrophic failure'), target_fixture="ctx")
def given_tier0_catastrophic_policy(cand: str):
    return {"candidate": cand, "control": "control", "forced_catastrophe": True}


@when("the Tier 0 mechanism smoke test runs on 8 fixed tasks")
def when_tier0_smoke_runs(ctx):
    rep = run_tier0_smoke_test(
        candidate_policy=ctx["candidate"],
        control_policy=ctx["control"],
        force_catastrophic_failure=ctx.get("forced_catastrophe", False),
    )
    ctx["smoke_report"] = rep


@then("8 tasks are evaluated across small edits, cross-file, test failures, large modules, ambiguous, and easy control")
def then_tier0_eight_tasks(ctx):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert len(rep.tasks) == 8
    task_types = {t["task_type"] for t in rep.tasks}
    assert "small/local edit" in task_types
    assert "cross-file" in task_types
    assert "test-driven failure" in task_types
    assert "large-module" in task_types
    assert "ambiguous/localization" in task_types
    assert "easy control" in task_types


@then("the smoke report records provider tokens, frontier turns, and verification success")
def then_tier0_metrics_recorded(ctx):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert rep.control.total_provider_tokens > 0
    assert rep.candidate.total_provider_tokens > 0
    assert rep.candidate.total_frontier_turns > 0
    assert rep.candidate.verification_success_count > 0


@then(parsers.parse('the delta solve is at least "{min_delta}"'))
def then_tier0_delta_solve(ctx, min_delta: str):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert rep.delta_solve >= int(min_delta)


@then(parsers.parse('the tokens per solved ratio is at most "{max_ratio}"'))
def then_tier0_tokens_ratio(ctx, max_ratio: str):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert rep.tokens_per_solved_ratio <= float(max_ratio)


@then("zero catastrophic regressions are detected")
def then_tier0_zero_catastrophes(ctx):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert rep.zero_catastrophic_regressions is True


@then(parsers.parse('the dev promotion gate verdict is "{verdict}"'))
def then_tier0_gate_verdict(ctx, verdict: str):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    expected = (verdict == "PASS")
    assert rep.passes_dev_gate is expected


@then(parsers.parse('the rejection reason mentions "{phrase}"'))
def then_tier0_rejection_reason(ctx, phrase: str):
    rep: Tier0SmokeReport = ctx["smoke_report"]
    assert phrase.lower() in rep.gate_reason.lower()


# --- MINTOK_CANARY = 12 Steps ---

@given("the MinTok canary suite configuration", target_fixture="canary_suite")
def given_canary_suite_cfg():
    return {"tasks": CANARY_TASKS, "count": MINTOK_CANARY}


@then("MINTOK_CANARY equals 12")
def then_mintok_canary_12(canary_suite):
    assert MINTOK_CANARY == 12
    assert len(CANARY_TASKS) == 12


@then("canary tasks cover large-module localization, class target, cross-file, failing test, and misleading traceback")
def then_canary_coverage(canary_suite):
    categories = {t["category"] for t in CANARY_TASKS}
    assert "large_module" in categories
    assert "small_local_edit" in categories
    assert "cross_file" in categories
    assert "test_driven_failure" in categories
    assert "traceback" in categories


# --- Tier 1 dev-20 Steps ---

@given(parsers.parse('benchmark suite "{suite}" with arms "{arms}"'), target_fixture="suite_cfg")
def given_suite_cfg(suite: str, arms: str):
    return {"suite": suite, "arms": arms}


@when("the paired mini-benchmark executes with interleaving enabled")
def when_suite_executes(suite_cfg):
    res = run_suite_benchmark(
        suite=suite_cfg["suite"],
        arms=suite_cfg["arms"],
        paired=True,
        interleaved=True,
        model="LOCAL_MODEL",
    )
    suite_cfg["result"] = res


@then("exactly 20 tasks are evaluated across stratified task families")
def then_suite_tasks_count(suite_cfg):
    res: PairedSuiteResult = suite_cfg["result"]
    assert res.tasks_count == 20
    assert len(res.tasks) == 20


@then(parsers.parse('execution order alternates between "{order1}" and "{order2}"'))
def then_suite_interleaved_order(suite_cfg, order1: str, order2: str):
    res: PairedSuiteResult = suite_cfg["result"]
    first_order = " -> ".join(res.tasks[0]["execution_order"])
    second_order = " -> ".join(res.tasks[1]["execution_order"])
    assert first_order == order1
    assert second_order == order2


@then(parsers.parse('the provider tokens ratio is at most "{max_ratio}"'))
def then_suite_tokens_ratio(suite_cfg, max_ratio: str):
    res: PairedSuiteResult = suite_cfg["result"]
    assert res.provider_tokens_ratio <= float(max_ratio)


@then(parsers.parse('the candidate solves at least {min_solves:d} tasks out of 20'))
def then_suite_min_solves(suite_cfg, min_solves: int):
    res: PairedSuiteResult = suite_cfg["result"]
    assert res.candidate_solved >= min_solves


@then("the report includes tokens per solved, p95 tokens, and verification counts")
def then_suite_report_fields(suite_cfg):
    res: PairedSuiteResult = suite_cfg["result"]
    d = res.to_dict()
    assert "tokens_per_solved" in d
    assert "p95_tokens" in d
    assert "verification" in d
    rendered = res.render_text()
    assert "tokens/solved:" in rendered
    assert "p95 tokens:" in rendered
    assert "verification:" in rendered


# --- Counterfactual Replay Steps ---

@given("historical trajectory records from previous runs", target_fixture="hist_records")
def given_historical_records():
    return [
        {"task_id": f"hist-task-{i}", "context_tokens": 15000 + i * 500, "has_large_tool_output": (i % 2 == 0)}
        for i in range(20)
    ]


@when(parsers.parse('the counterfactual replay engine simulates policy "{policy}"'))
def when_counterfactual_replay_runs(hist_records, pytestconfig):
    rep = run_counterfactual_replay(
        raw_records=hist_records,
        policy="v3_candidate",
    )
    hist_records.clear()
    hist_records.append(rep)


@then(parsers.parse('the context token reduction ratio is at least "{min_ratio}"'))
def then_replay_reduction_ratio(hist_records, min_ratio: str):
    rep: CounterfactualReplayReport = hist_records[0]
    assert rep.reduction_ratio >= float(min_ratio)


@then(parsers.parse('recovered observations exceed {min_rec:d} percent'))
def then_replay_recovered_obs(hist_records, min_rec: int):
    rep: CounterfactualReplayReport = hist_records[0]
    assert rep.recovered_observations_pct >= float(min_rec)


@then(parsers.parse('predicted lost evidence is below {max_lost:d} percent'))
def then_replay_lost_evidence(hist_records, max_lost: int):
    rep: CounterfactualReplayReport = hist_records[0]
    assert rep.predicted_lost_evidence_pct <= float(max_lost)


@then("tasks are partitioned into definitely unaffected and potentially affected sets")
def then_replay_partition(hist_records):
    rep: CounterfactualReplayReport = hist_records[0]
    assert rep.tasks_unaffected_count > 0
    assert rep.tasks_potentially_affected_count > 0
    assert rep.tasks_unaffected_count + rep.tasks_potentially_affected_count == rep.tasks_evaluated


# --- Policy Predictor Steps ---

@given("an agent state with active failures and high token consumption", target_fixture="agent_state")
def given_agent_state():
    return AgentState(
        turn=8,
        tokens_spent=65000,
        active_failures=2,
        patch_lines=15,
        repo_complexity=0.75,
        task_family="large_module",
        context_window_usage=0.80,
    )


@given(parsers.parse('candidate actions including "{a1}", "{a2}", "{a3}", and "{a4}"'), target_fixture="candidate_actions")
def given_candidate_actions(a1: str, a2: str, a3: str, a4: str):
    return [
        CandidateAction(action_type=a1, target="full_source.py", raw_tokens=18000),
        CandidateAction(action_type=a2, target="slice:lines 40-80", raw_tokens=4000),
        CandidateAction(action_type=a3, target="pytest tests/test_core.py", raw_tokens=6000),
        CandidateAction(action_type=a4, target="compact", raw_tokens=2000),
    ]


@when(parsers.parse('the policy predictor scores candidate actions with solve value {val:f} and token lambda {lmb:f}'))
def when_policy_predictor_scores(agent_state, candidate_actions, val: float, lmb: float, pytestconfig):
    pred = PolicyPredictor(solve_value=val, token_lambda=lmb)
    ranked = pred.rank_actions(agent_state, candidate_actions)
    pytestconfig._ranked_actions = ranked


@then("every candidate action receives a P(solve), expected tokens, and utility score")
def then_predictor_action_scores(pytestconfig):
    ranked = pytestconfig._ranked_actions
    assert len(ranked) >= 4
    for r in ranked:
        assert 0.0 < r.p_solve < 1.0
        assert r.expected_tokens > 0
        assert isinstance(r.utility_score, float)


@then(parsers.parse('"{act1}" or "{act2}" achieves higher utility than "{act3}"'))
def then_predictor_utility_hierarchy(pytestconfig, act1: str, act2: str, act3: str):
    ranked = pytestconfig._ranked_actions
    scores_by_act = {r.action.action_type: r.utility_score for r in ranked}
    assert (
        scores_by_act[act1] > scores_by_act[act3]
        or scores_by_act[act2] > scores_by_act[act3]
    )


# --- Content-Addressed Model Caching Steps ---

@given("an inference setup with model, system prompt, task, repo snapshot, and tools", target_fixture="inference_setup")
def given_inference_setup():
    return {
        "model": "qwen2.5-coder-7b",
        "model_snapshot": "2026-09-01",
        "system_prompt": "You are a coding agent.",
        "task": "Fix bug in calculator",
        "repo_snapshot": "commit_abc123",
        "tools": "read,edit,test",
        "temperature": 0.0,
        "reasoning_effort": "none",
    }


@when("the content-addressed cache key is computed")
def when_cache_key_computed(inference_setup, pytestconfig):
    from mintok.execution_harness import LocalModelCache

    key = LocalModelCache.compute_content_addressed_key(
        model=inference_setup["model"],
        model_snapshot=inference_setup["model_snapshot"],
        system_prompt=inference_setup["system_prompt"],
        task=inference_setup["task"],
        repo_snapshot=inference_setup["repo_snapshot"],
        tools=inference_setup["tools"],
        temperature=inference_setup["temperature"],
        reasoning_effort=inference_setup["reasoning_effort"],
    )
    pytestconfig._content_key = key
    cache = LocalModelCache()
    cache.put(key, {"solution": "fixed", "tokens": 450})
    pytestconfig._model_cache = cache


@then("the key matches the SHA256 digest of the complete configuration tuple")
def then_key_matches_sha256(inference_setup, pytestconfig):
    import hashlib

    raw = f"{inference_setup['model']}:{inference_setup['model_snapshot']}:{inference_setup['system_prompt']}:{inference_setup['task']}:{inference_setup['repo_snapshot']}:{inference_setup['tools']}:{inference_setup['temperature']}:{inference_setup['reasoning_effort']}"
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert pytestconfig._content_key == expected


@then("querying the cache with identical configuration requires zero model calls")
def then_zero_model_calls(pytestconfig):
    cache = pytestconfig._model_cache
    calls_made = 0

    def dummy_model():
        nonlocal calls_made
        calls_made += 1
        return {"solution": "fresh_call"}

    result = cache.get(pytestconfig._content_key)
    assert result is not None
    assert result["solution"] == "fixed"
    assert calls_made == 0


# --- Stratified Fast Set & Checker Tiers Steps ---

@given("the frozen 12-task fast set specification", target_fixture="fast_set")
def given_frozen_fast_set():
    from mintok.fast_window import FAST12_STRATIFIED_FROZEN_SET

    return FAST12_STRATIFIED_FROZEN_SET


@then("exactly 12 tasks are included")
def then_exactly_12_tasks(fast_set):
    assert len(fast_set) == 12


@then("the set contains 2 small localized tasks, 2 cross-file tasks, 2 large-module tasks, 2 test debugging tasks, 2 API schema tasks, and 2 difficult failure cases")
def then_stratified_categories_present(fast_set):
    categories = [t[2] for t in fast_set]
    assert categories.count("small_localized") == 2
    assert categories.count("cross_file") == 2
    assert categories.count("large_module") == 2
    assert categories.count("test_debugging") == 2
    assert categories.count("api_schema") == 2
    assert categories.count("difficult_failure_case") == 2


@given(parsers.parse('the three checker tiers "{t1}", "{t2}", and "{t3}"'))
def given_checker_tiers(t1: str, t2: str, t3: str):
    from mintok.fast_window import CheckerLevel

    assert CheckerLevel.FAST == t1
    assert CheckerLevel.DEV == t2
    assert CheckerLevel.RELEASE == t3


@then(parsers.parse('"{t}" specifies {count:d} tasks with a turn cap of {cap:d}'))
def then_checker_fast_tier(t: str, count: int, cap: int):
    assert count == 12
    assert cap == 12


@then(parsers.parse('"{t}" specifies {count:d} tasks with full checker'))
def then_checker_dev_tier(t: str, count: int):
    assert count == 25


@then(parsers.parse('"{t}" specifies up to {count:d} tasks with frozen environment and full accounting'))
def then_checker_release_tier(t: str, count: int):
    assert count == 200


# --- Stop-Loss Controller Steps ---

@given("a stop-loss controller tracking patch attempts", target_fixture="stop_loss_ctx")
def given_stop_loss_patch_tracker():
    from mintok.stop_loss import StopLossController

    return {"controller": StopLossController()}


@when("3 consecutive patches fail without evidence gathering")
def when_three_patches_fail(stop_loss_ctx):
    ctrl = stop_loss_ctx["controller"]
    ctrl.record_patch_attempt(success=False, failure_signature="AssertionError: 4 != 5")
    ctrl.record_patch_attempt(success=False, failure_signature="AssertionError: 4 != 5")
    ctrl.record_patch_attempt(success=False, failure_signature="AssertionError: 4 != 5")


@then(parsers.parse('a proposed "{action}" action is intercepted with verdict "{verdict}"'))
def then_patch_intercepted(stop_loss_ctx, action: str, verdict: str):
    ctrl = stop_loss_ctx["controller"]
    sl_action, reason = ctrl.evaluate_proposed_action(action)
    assert sl_action.value == verdict
    stop_loss_ctx["reason"] = reason


@then("the controller demands investigation or testing before next edit")
def then_demands_investigation(stop_loss_ctx):
    reason = stop_loss_ctx["reason"].lower()
    assert "investigation" in reason or "slice" in reason or "test" in reason


@given("a stop-loss controller tracking observation handles", target_fixture="stop_loss_ctx")
def given_stop_loss_expansion_tracker():
    from mintok.stop_loss import StopLossController

    return {"controller": StopLossController()}


@when(parsers.parse('observation "{obs_id}" is expanded {count:d} times'))
def when_obs_expanded(stop_loss_ctx, obs_id: str, count: int):
    ctrl = stop_loss_ctx["controller"]
    for _ in range(count):
        ctrl.record_observation_expansion(obs_id)


@then(parsers.parse('a proposed "{action}" action on "{obs_id}" is intercepted with verdict "{verdict}"'))
def then_expansion_intercepted(stop_loss_ctx, action: str, obs_id: str, verdict: str):
    ctrl = stop_loss_ctx["controller"]
    sl_action, reason = ctrl.evaluate_proposed_action(action, target=obs_id)
    assert sl_action.value == verdict


# --- Evidence Sufficiency Steps ---

@given("evidence sufficiency state with target, caller, failure, patch location, and expected behavior known", target_fixture="controller_ctx")
def given_sufficiency_state():
    from mintok.information_allocator import EvidenceSufficiencyState
    from mintok.mintok_controller import MinTokController

    sufficiency = EvidenceSufficiencyState(
        target_known=True,
        caller_known=True,
        failure_known=True,
        patch_location_known=True,
        expected_behavior_known=True,
    )
    ctrl = MinTokController(sufficiency=sufficiency)
    return {"controller": ctrl}


@when(parsers.parse('the agent proposes an expensive "{action}" action'))
def when_agent_proposes_expensive_read(controller_ctx, action: str):
    from mintok.predictor import AgentState

    ctrl = controller_ctx["controller"]
    state = AgentState()
    decision = ctrl.process_agent_step(state, requested_action=action, target="module_large.py", raw_tokens=8000)
    controller_ctx["decision"] = decision


@then(parsers.parse('the MinTok controller intercepts the read and directs the agent to "{target_action}"'))
def then_controller_redirects_to_patch(controller_ctx, target_action: str):
    decision = controller_ctx["decision"]
    assert decision.intercepted is True
    assert decision.allocated_action == target_action


# --- Adaptive Verifier Steps ---

@given("the adaptive verifier", target_fixture="verifier_inst")
def given_adaptive_verifier():
    from mintok.information_allocator import AdaptiveVerifier

    return AdaptiveVerifier()


@when("blast radius is evaluated for a local function change")
def when_eval_local_blast(verifier_inst, pytestconfig):
    from mintok.information_allocator import BlastRadius

    radius = verifier_inst.estimate_blast_radius(
        files_modified=["src/calc.py"],
        symbols_modified=["calc.add"],
        is_public_api_modified=False,
    )
    assert radius == BlastRadius.LOCAL_FUNCTION
    dec = verifier_inst.select_verification(radius)
    pytestconfig._verif_decision = dec


@then(parsers.parse('the verifier selects "{scope}" scope with an estimated token budget below {budget:d}'))
def then_verif_scope_budget(pytestconfig, scope: str, budget: int):
    dec = pytestconfig._verif_decision
    assert dec.recommended_scope == scope
    assert dec.estimated_tokens < budget


@when("blast radius is evaluated for a public API signature modification")
def when_eval_api_blast(verifier_inst, pytestconfig):
    from mintok.information_allocator import BlastRadius

    radius = verifier_inst.estimate_blast_radius(
        files_modified=["src/api.py"],
        symbols_modified=["api.charge"],
        is_public_api_modified=True,
    )
    assert radius == BlastRadius.PUBLIC_API
    dec = verifier_inst.select_verification(radius)
    pytestconfig._verif_decision = dec


@then(parsers.parse('the verifier selects "{scope}" scope'))
def then_verif_scope(pytestconfig, scope: str):
    dec = pytestconfig._verif_decision
    assert dec.recommended_scope == scope


# --- Local Action Critic Steps ---

@given("the local action critic and a state with a pending unverified patch", target_fixture="critic_ctx")
def given_critic_unverified_patch():
    from mintok.information_allocator import LocalActionCritic

    critic = LocalActionCritic()
    return {"critic": critic}


@when(parsers.parse('the agent proposes another "{action}" action'))
def when_agent_proposes_patch(critic_ctx, action: str):
    critic = critic_ctx["critic"]
    verdict, reason = critic.evaluate(
        proposed_action=action,
        target="calc.py",
        known_read_targets=[],
        has_pending_patch=True,
    )
    critic_ctx["verdict"] = verdict
    critic_ctx["reason"] = reason


@then(parsers.parse('the critic classifies the action as "{verdict}"'))
def then_critic_verdict(critic_ctx, verdict: str):
    assert critic_ctx["verdict"].value == verdict


# --- Champion Challenger Steps ---

@given(parsers.parse('incumbent champion with {solves:d} solves and {tokens:d} tokens on the fast set'), target_fixture="champ_ctx")
def given_champion_stats(solves: int, tokens: int):
    return {"champ_solves": solves, "champ_tokens": tokens}


@when(parsers.parse('challenger policy achieves {solves:d} solves and {tokens:d} tokens'))
def when_challenger_achieves(champ_ctx, solves: int, tokens: int):
    from mintok.mintok_controller import ChampionChallengerTournament

    res = ChampionChallengerTournament.evaluate(
        champion_policy="v3_champion",
        challenger_policy="v3_challenger",
        champion_solves=champ_ctx["champ_solves"],
        challenger_solves=solves,
        champion_tokens=champ_ctx["champ_tokens"],
        challenger_tokens=tokens,
    )
    champ_ctx["result"] = res


@then("the tournament verdict is promoted with Pareto improvement confirmed")
def then_tournament_promoted(champ_ctx):
    res = champ_ctx["result"]
    assert res.promoted is True
    assert res.pareto_improved is True


@when(parsers.parse('another challenger achieves {solves:d} solves and {tokens:d} tokens'))
def when_another_challenger_achieves(champ_ctx, solves: int, tokens: int):
    from mintok.mintok_controller import ChampionChallengerTournament

    res = ChampionChallengerTournament.evaluate(
        champion_policy="v3_champion",
        challenger_policy="v3_regressed",
        champion_solves=champ_ctx["champ_solves"],
        challenger_solves=solves,
        champion_tokens=champ_ctx["champ_tokens"],
        challenger_tokens=tokens,
    )
    champ_ctx["result_regressed"] = res


@then("the tournament verdict is rejected due to solve regression")
def then_tournament_rejected(champ_ctx):
    res = champ_ctx["result_regressed"]
    assert res.promoted is False
    assert "regressed" in res.reason.lower()


# --- Hypothesis Graph Steps ---

@given(parsers.parse('a hypothesis graph with root "{root}" and child "{child}"'), target_fixture="hyp_ctx")
def given_hypothesis_tree(root: str, child: str):
    from mintok.hypothesis import HypothesisGraph

    g = HypothesisGraph()
    g.add_hypothesis("h_root", root, confidence=0.7)
    g.add_hypothesis("h_child", child, confidence=0.5, parent_id="h_root", discriminating_features={"vat_exemptions": True})
    return {"graph": g, "root_id": "h_root", "child_id": "h_child"}


@when("evidence is observed that VAT exemptions are disabled in config")
def when_evidence_observed(hyp_ctx):
    g = hyp_ctx["graph"]
    hyp_ctx["initial_entropy"] = g.compute_entropy()
    g.update_confidence(hyp_ctx["child_id"], likelihood_ratio=0.05, evidence="Config shows VAT exemption disabled", supports=False)


@then("the child hypothesis confidence updates and Shannon entropy is reduced")
def then_hyp_confidence_and_entropy(hyp_ctx):
    g = hyp_ctx["graph"]
    child_node = g.nodes[hyp_ctx["child_id"]]
    assert child_node.confidence < 0.20
    assert len(child_node.refuting_evidence) == 1
    new_entropy = g.compute_entropy()
    assert new_entropy <= hyp_ctx["initial_entropy"]


# --- Negative Evidence Memory Steps ---

@given(parsers.parse('negative evidence memory with verified fact "{fact_key}" for target "{target}"'), target_fixture="neg_ev_ctx")
def given_neg_evidence(fact_key: str, target: str):
    from mintok.hypothesis import NegativeEvidenceMemory
    from mintok.mintok_controller import MinTokController

    nem = NegativeEvidenceMemory()
    nem.record_negative_fact(fact_key=fact_key, target=target, rationale="Static AST scan verified no timeout writes")
    ctrl = MinTokController(negative_evidence=nem)
    return {"memory": nem, "controller": ctrl, "target": target}


@when(parsers.parse('the agent proposes reading or querying "{target}" for timeout writes'))
def when_agent_queries_ruled_out_target(neg_ev_ctx, target: str):
    from mintok.predictor import AgentState

    ctrl = neg_ev_ctx["controller"]
    state = AgentState()
    dec = ctrl.process_agent_step(state, requested_action="read_slice", target=target)
    neg_ev_ctx["decision"] = dec


@then("the MinTok controller intercepts the action citing the negative evidence memory")
def then_intercepted_by_negative_evidence(neg_ev_ctx):
    dec = neg_ev_ctx["decision"]
    assert dec.intercepted is True
    assert "negative evidence" in dec.interception_reason.lower()


# --- Observation Dependency & Surgical Invalidation Steps ---

@given(parsers.parse('a cached observation for "{file_path}" covering lines {start:d} to {end:d}'), target_fixture="obs_cache_ctx")
def given_obs_cache_span1(file_path: str, start: int, end: int):
    from mintok.observation_cache import ObservationDependencyGraph, SymbolSpan, TTLMode

    graph = ObservationDependencyGraph()
    obs1 = graph.register_observation(
        observation_id="obs_span_100_150",
        observation_type="slice",
        content_summary="Tax calculation function",
        files_covered=[file_path],
        spans=[SymbolSpan(file_path=file_path, start_line=start, end_line=end, symbol_name="calc_tax")],
        ttl_mode=TTLMode.UNTIL_CODE_CHANGE,
        token_size=800,
    )
    return {"graph": graph, "obs1": obs1, "file_path": file_path}


@given(parsers.parse('another cached observation covering lines {start:d} to {end:d}'))
def given_obs_cache_span2(obs_cache_ctx, start: int, end: int):
    from mintok.observation_cache import SymbolSpan, TTLMode

    graph = obs_cache_ctx["graph"]
    file_path = obs_cache_ctx["file_path"]
    obs2 = graph.register_observation(
        observation_id="obs_span_10_30",
        observation_type="slice",
        content_summary="Imports and header config",
        files_covered=[file_path],
        spans=[SymbolSpan(file_path=file_path, start_line=start, end_line=end, symbol_name="config")],
        ttl_mode=TTLMode.UNTIL_CODE_CHANGE,
        token_size=400,
    )
    obs_cache_ctx["obs2"] = obs2


@when(parsers.parse('a patch hunk modifies "{file_path}" from line {start:d} to line {end:d}'))
def when_patch_hunk_applied(obs_cache_ctx, file_path: str, start: int, end: int):
    graph = obs_cache_ctx["graph"]
    res = graph.invalidate_on_patch(file_path=file_path, patch_start_line=start, patch_end_line=end)
    obs_cache_ctx["invalidation_res"] = res


@then(parsers.parse('the observation covering lines {start:d} to {end:d} is invalidated'))
def then_obs_invalidated(obs_cache_ctx, start: int, end: int):
    res = obs_cache_ctx["invalidation_res"]
    assert "obs_span_10_30" in res.invalidated_ids
    obs2 = obs_cache_ctx["obs2"]
    assert obs2.is_valid is False


@then(parsers.parse('the observation covering lines {start:d} to {end:d} is retained with its context tokens intact'))
def then_obs_retained(obs_cache_ctx, start: int, end: int):
    res = obs_cache_ctx["invalidation_res"]
    assert "obs_span_100_150" in res.retained_ids
    obs1 = obs_cache_ctx["obs1"]
    assert obs1.is_valid is True
    assert res.retained_tokens >= 800


# --- Observation Cache Tiered TTL Steps ---

@given(parsers.parse('observations with TTL modes "{t1}", "{t2}", and "{t3}"'), target_fixture="ttl_ctx")
def given_ttl_modes(t1: str, t2: str, t3: str):
    from mintok.observation_cache import ObservationDependencyGraph, TTLMode

    g = ObservationDependencyGraph()
    o_run = g.register_observation("o_run", "info", "Architecture", ["src/main.py"], ttl_mode=TTLMode.RUN, turn=1, token_size=200)
    o_patch = g.register_observation("o_patch", "test", "Pytest output", ["tests/test.py"], ttl_mode=TTLMode.UNTIL_PATCH, turn=1, token_size=300)
    o_imm = g.register_observation("o_imm", "cmd", "ls output", ["."], ttl_mode=TTLMode.IMMEDIATE, turn=1, token_size=100)
    return {"graph": g, "o_run": o_run, "o_patch": o_patch, "o_imm": o_imm}


@when(parsers.parse('a turn advances, "{imm_mode}" observations are expired'))
def when_turn_advances_expire_imm(ttl_ctx, imm_mode: str):
    g = ttl_ctx["graph"]
    inv = g.advance_turn(new_turn=2)
    assert "o_imm" in inv
    assert ttl_ctx["o_imm"].is_valid is False


@when(parsers.parse('a code patch is applied, "{patch_mode}" observations are invalidated'))
def when_patch_applied_expire_patch(ttl_ctx, patch_mode: str):
    g = ttl_ctx["graph"]
    res = g.invalidate_on_patch(file_path="src/main.py", patch_start_line=1, patch_end_line=5)
    assert "o_patch" in res.invalidated_ids
    assert ttl_ctx["o_patch"].is_valid is False


@then(parsers.parse('"{run_mode}" observations remain valid throughout the entire session'))
def then_run_mode_retained(ttl_ctx, run_mode: str):
    assert ttl_ctx["o_run"].is_valid is True


# --- Action Value Model Steps ---

@given(parsers.parse('an action value model and candidate actions "{a1}" and "{a2}"'), target_fixture="act_val_ctx")
def given_act_val_model(a1: str, a2: str):
    from mintok.action_value import ActionValueModel

    return {"model": ActionValueModel(), "a1": a1, "a2": a2}


@when("the observation value is evaluated")
def when_eval_observation_val(act_val_ctx):
    m = act_val_ctx["model"]
    v1 = m.evaluate_observation_value(action_type=act_val_ctx["a1"], target="billing/calc.py", base_tokens=2500)
    v2 = m.evaluate_observation_value(action_type=act_val_ctx["a2"], target="billing/calc.py", base_tokens=2500)
    act_val_ctx["v1"] = v1
    act_val_ctx["v2"] = v2


@then(parsers.parse('"{a1}" yields higher net observation value than "{a2}" by saving future search tokens'))
def then_net_obs_val_higher(act_val_ctx, a1: str, a2: str):
    v1 = act_val_ctx["v1"]
    v2 = act_val_ctx["v2"]
    assert v1.net_observation_value > v2.net_observation_value
    assert v1.estimated_future_tokens_saved > 0


# --- Propensity Logging & Rejected Alternatives Steps ---

@given("the MinTok controller processing agent steps with active exploration", target_fixture="prop_ctx")
def given_propensity_controller():
    from mintok.mintok_controller import MinTokController

    ctrl = MinTokController()
    return {"controller": ctrl}


@when("decisions are evaluated across candidate actions")
def when_propensity_decisions_evaluated(prop_ctx):
    from mintok.predictor import AgentState

    ctrl = prop_ctx["controller"]
    state = AgentState(turn=1, tokens_spent=4000)
    dec = ctrl.process_agent_step(state, requested_action="read_slice", target="billing/calc.py", raw_tokens=2000)
    prop_ctx["decision"] = dec


@then("every decision records an action propensity distribution summing to 1.0")
def then_propensity_sums_to_one(prop_ctx):
    dec = prop_ctx["decision"]
    dist = dec.propensity_distribution
    assert len(dist) >= 2
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-4


@then("the chosen action and all rejected alternatives are logged with their utility scores")
def then_rejected_alternatives_logged(prop_ctx):
    ctrl = prop_ctx["controller"]
    recs = ctrl.alternatives_logger.get_records()
    assert len(recs) >= 1
    last_rec = recs[-1]
    assert last_rec.chosen_action == "read_slice"
    assert len(last_rec.rejected_alternatives) >= 2
    for rej in last_rec.rejected_alternatives:
        assert rej.action_type != ""
        assert isinstance(rej.utility_score, float)


# --- Off-Policy Evaluation Steps ---

@given("logged trajectory propensity records from a historical run", target_fixture="ope_ctx")
def given_logged_propensities():
    from mintok.propensity import PropensityRecord

    records = []
    for i in range(25):
        records.append(
            PropensityRecord(
                state_turn=i + 1,
                state_features=[float(i) / 10.0, 0.5, 0.2],
                candidate_actions=["read_slice", "read_full", "run_verifier"],
                candidate_utilities={"read_slice": 0.8, "read_full": 0.3, "run_verifier": 0.6},
                propensity_distribution={"read_slice": 0.85, "read_full": 0.05, "run_verifier": 0.10},
                chosen_action="read_slice" if i % 4 != 0 else "run_verifier",
                is_exploration=(i % 4 == 0),
                logging_propensity=0.85 if i % 4 != 0 else 0.10,
                actual_tokens=650 if i % 4 != 0 else 1200,
                observed_reward=1.0 if i % 5 != 0 else 0.0,
            )
        )
    return {"records": records}


@when("off-policy evaluation simulates a candidate context allocation policy")
def when_ope_simulates(ope_ctx):
    from mintok.propensity import OffPolicyEvaluator

    def target_policy(features, candidates):
        return {"read_slice": 0.90, "read_full": 0.02, "run_verifier": 0.08}

    res = OffPolicyEvaluator.evaluate(records=ope_ctx["records"], target_policy_fn=target_policy)
    ope_ctx["result"] = res


@then("importance sampling and weighted importance sampling rewards are computed with effective sample size")
def then_ope_metrics_computed(ope_ctx):
    res = ope_ctx["result"]
    assert res.sample_size == 25
    assert res.importance_sampling_reward > 0.0
    assert res.weighted_importance_sampling_reward > 0.0
    assert res.effective_sample_size > 0.0


# --- Survival Predictor Steps ---

@given(parsers.parse('an agent trajectory with {failed:d} failed patches, {tokens:d} tokens spent, and repeated actions'), target_fixture="surv_ctx")
def given_surv_trajectory(failed: int, tokens: int):
    return {"failed": failed, "tokens": tokens, "turn": 7, "repeats": 2}


@when("survival analysis evaluates the trajectory state")
def when_eval_survival(surv_ctx):
    from mintok.action_value import SurvivalPredictor

    p_solve, hazard, rec = SurvivalPredictor.evaluate(
        turn=surv_ctx["turn"],
        tokens_spent=surv_ctx["tokens"],
        failed_patches=surv_ctx["failed"],
        repeated_actions=surv_ctx["repeats"],
    )
    surv_ctx["p_solve"] = p_solve
    surv_ctx["hazard"] = hazard
    surv_ctx["recommendation"] = rec


@then(parsers.parse('the runaway hazard exceeds {threshold:f} and the recommendation is "{rec1}" or "{rec2}"'))
def then_surv_hazard_exceeds(surv_ctx, threshold: float, rec1: str, rec2: str):
    assert surv_ctx["hazard"] >= threshold
    assert surv_ctx["recommendation"] in (rec1, rec2)


# --- Trajectory Fingerprinter Steps ---

@given(parsers.parse('an agent trajectory with {reads:d} consecutive reads without editing and {tokens:d} tokens spent'), target_fixture="fingerprint_ctx")
def given_fingerprint_seq(reads: int, tokens: int):
    seq = ["read_full"] * reads
    return {"sequence": seq, "tokens": tokens}


@when("trajectory fingerprinting classifies the behavior")
def when_classify_fingerprint(fingerprint_ctx):
    from mintok.action_value import TrajectoryFingerprinter

    arch, conf, interv = TrajectoryFingerprinter.classify(
        action_sequence=fingerprint_ctx["sequence"],
        tokens_spent=fingerprint_ctx["tokens"],
        failed_patches=0,
        verification_count=0,
    )
    fingerprint_ctx["archetype"] = arch
    fingerprint_ctx["confidence"] = conf
    fingerprint_ctx["intervention"] = interv


@then(parsers.parse('the archetype is identified as "{expected_arch}" with an intervention recommending token budgeting'))
def then_fingerprint_matches(fingerprint_ctx, expected_arch: str):
    assert fingerprint_ctx["archetype"].value == expected_arch
    assert fingerprint_ctx["confidence"] >= 0.70
    assert "token" in fingerprint_ctx["intervention"].lower() or "budget" in fingerprint_ctx["intervention"].lower()


# --- Pre-Action Token Budgeting Steps ---

@given(parsers.parse('a pre-action token budget of {budget:d} tokens'), target_fixture="budget_ctx")
def given_action_budget(budget: int):
    return {"budget": budget}


@when(parsers.parse('the budget negotiator formats a "{action}" action'))
def when_negotiate_budget(budget_ctx, action: str):
    from mintok.token_budgeter import ActionBudgetNegotiator

    spec = ActionBudgetNegotiator.negotiate_budget(action_type=action, target="billing/calc.py", budget=budget_ctx["budget"])
    budget_ctx["spec"] = spec


@then(parsers.parse('the action is configured with "{fmt}" format and CALL_GRAPH precision within budget'))
def then_action_configured(budget_ctx, fmt: str):
    from mintok.token_budgeter import PrecisionLevel

    spec = budget_ctx["spec"]
    assert spec.content_format == fmt
    assert spec.precision_level == PrecisionLevel.CALL_GRAPH
    assert spec.estimated_tokens <= budget_ctx["budget"] + 100


# --- Prompt-Cache Economics Steps ---

@given(parsers.parse('existing cached context of {tokens:d} tokens'), target_fixture="pce_ctx")
def given_existing_cached_tokens(tokens: int):
    return {"cached_tokens": tokens}


@when(parsers.parse('comparing appending {appended:d} tokens versus rewriting the entire context'))
def when_eval_append_vs_rewrite(pce_ctx, appended: int):
    from mintok.token_budgeter import PromptCacheEconomics

    pce = PromptCacheEconomics()
    app_cost, rew_cost = pce.evaluate_append_vs_rewrite(
        existing_cached_tokens=pce_ctx["cached_tokens"],
        appended_tokens=appended,
        output_tokens=300,
    )
    pce_ctx["app_cost"] = app_cost
    pce_ctx["rew_cost"] = rew_cost


@then("appending preserves the prompt-cache prefix and costs substantially less than rewriting")
def then_append_costs_less(pce_ctx):
    app = pce_ctx["app_cost"]
    rew = pce_ctx["rew_cost"]
    assert app.cache_hit_ratio > 0.85
    assert rew.cache_hit_ratio == 0.0
    assert app.cost_dollars < (rew.cost_dollars * 0.40)


# --- Trajectory State Machine & ACT_NOW Steps ---

@given(parsers.parse('a trajectory state machine in phase "{phase}"'), target_fixture="sm_ctx")
def given_state_machine_phase(phase: str):
    from mintok.trajectory_sm import TrajectoryPhase, TrajectoryStateMachine

    sm = TrajectoryStateMachine(initial_phase=TrajectoryPhase(phase))
    return {"sm": sm}


@when("target is known, fault is localized, and expected behavior is confirmed")
def when_milestones_reached(sm_ctx):
    sm = sm_ctx["sm"]
    phase = sm.evaluate_transition(
        turn=3,
        target_known=True,
        fault_localized=True,
        hypothesis_confirmed=True,
        patch_applied=False,
        verification_passed=False,
        recent_failure=False,
    )
    sm_ctx["transitioned_phase"] = phase
    sm_ctx["act_now"] = sm.should_act_now(
        target_known=True,
        patch_location_known=True,
        expected_behavior_known=True,
    )


@then(parsers.parse('the state machine transitions to phase "{expected_phase}"'))
def then_sm_transitions(sm_ctx, expected_phase: str):
    assert sm_ctx["transitioned_phase"].value == expected_phase


@then("ACT_NOW value of waiting confirms the agent should execute immediately")
def then_act_now_confirmed(sm_ctx):
    assert sm_ctx["act_now"] is True


# --- Macro-Planner & Tool Complementarity Steps ---

@given(parsers.parse('a macro-planner generating sequences for phase "{phase}"'), target_fixture="macro_ctx")
def given_macro_planner(phase: str):
    from mintok.trajectory_sm import TrajectoryPhase

    return {"phase": TrajectoryPhase(phase)}


@when(parsers.parse('a macro-plan is generated with steps "{s1}", "{s2}", and "{s3}"'))
def when_macro_plan_generated(macro_ctx, s1: str, s2: str, s3: str):
    from mintok.trajectory_sm import MacroPlanner

    plan = MacroPlanner.generate_plan(phase=macro_ctx["phase"], target="billing/calc.py")
    macro_ctx["plan"] = plan


@then(parsers.parse('the plan contains exactly {count:d} steps within budget'))
def then_plan_steps_count(macro_ctx, count: int):
    plan = macro_ctx["plan"]
    assert len(plan.steps) == count
    assert plan.estimated_total_tokens <= 2000


@then("the tool complementarity score is strictly positive due to pairwise synergy")
def then_complementarity_positive(macro_ctx):
    plan = macro_ctx["plan"]
    assert plan.complementarity_score > 0.0


# --- Minimum Sufficient Patch Predictor Steps ---

@given(parsers.parse('a bugfix task expecting a 1-file 8-line modification'), target_fixture="patch_pred_ctx")
def given_patch_prediction():
    from mintok.patch_risk import MinimumSufficientPatchPredictor

    exp = MinimumSufficientPatchPredictor.predict_expectation(task_family="bugfix")
    return {"expectation": exp}


@when(parsers.parse('an agent proposes a bloated patch touching {files:d} files and {lines:d} lines'))
def when_bloated_patch_proposed(patch_pred_ctx, files: int, lines: int):
    from mintok.patch_risk import MinimumSufficientPatchPredictor

    verdict = MinimumSufficientPatchPredictor.evaluate_proposed_patch(
        expectation=patch_pred_ctx["expectation"],
        actual_files_count=files,
        actual_lines_count=lines,
    )
    patch_pred_ctx["verdict"] = verdict


@then(parsers.parse('the patch anomaly verdict is "{rec}" with bloat ratio exceeding {min_bloat:f}'))
def then_patch_anomaly_verdict(patch_pred_ctx, rec: str, min_bloat: float):
    verdict = patch_pred_ctx["verdict"]
    assert verdict.is_anomaly is True
    assert verdict.recommendation == rec
    assert verdict.bloat_ratio >= min_bloat


# --- Token Arbitrage & Marginal Intelligence Steps ---

@given(parsers.parse('model options "{m1}" with {p1:f} solve at ${c1:f} and "{m2}" with {p2:f} solve at ${c2:f}'), target_fixture="arb_ctx")
def given_model_options(m1: str, p1: float, c1: float, m2: str, p2: float, c2: float):
    from mintok.token_arbitrage import ModelCandidate, ModelTier

    opt1 = ModelCandidate(model_name=m1, tier=ModelTier.CHEAP_API, p_solve=p1, token_cost=5000, dollar_cost=c1)
    opt2 = ModelCandidate(model_name=m2, tier=ModelTier.FRONTIER, p_solve=p2, token_cost=20000, dollar_cost=c2)
    return {"opt1": opt1, "opt2": opt2}


@when("marginal intelligence escalation is evaluated")
def when_eval_marginal_intelligence(arb_ctx):
    from mintok.token_arbitrage import MarginalIntelligenceEvaluator

    res = MarginalIntelligenceEvaluator.evaluate_escalation(arb_ctx["opt1"], arb_ctx["opt2"])
    arb_ctx["result"] = res


@then(parsers.parse('the escalation to frontier is rejected because +0.01 solve does not justify the 4x cost delta'))
def then_escalation_rejected(arb_ctx):
    res = arb_ctx["result"]
    assert res.rejected_escalation is True
    assert res.marginal_gain <= 0.02


@then(parsers.parse('"{expected_model}" is selected as the winning execution tier'))
def then_selected_tier(arb_ctx, expected_model: str):
    assert arb_ctx["result"].selected_model == expected_model


# --- Dynamic Oracle & Avoidable Spend Steps ---

@given(parsers.parse('an agent trajectory spending {tokens:d} tokens on a 50-line target'), target_fixture="dyn_oracle_ctx")
def given_trajectory_spending(tokens: int):
    return {"tokens": tokens, "target_loc": 50}


@when("the dynamic oracle and avoidable spend breakdown are computed")
def when_compute_dynamic_oracle(dyn_oracle_ctx):
    from mintok.patch_risk import InferenceMilestoneTracker
    from mintok.token_arbitrage import DynamicOracle

    oracle = DynamicOracle.compute(task_id="task_calc", actual_tokens_spent=dyn_oracle_ctx["tokens"], target_loc=50, test_loc=40)
    tracker = InferenceMilestoneTracker()
    tracker.record_step(step_tokens=dyn_oracle_ctx["tokens"], is_correct_hypothesis=True)
    breakdown = tracker.compute_avoidable_breakdown(
        oracle_tokens=oracle.oracle_total_tokens,
        redundant_reads_count=3,
        failed_patches_count=2,
        excess_verification_tokens=4000,
    )
    dyn_oracle_ctx["oracle"] = oracle
    dyn_oracle_ctx["breakdown"] = breakdown


@then(parsers.parse('the dynamic oracle specifies less than {max_oracle:d} tokens'))
def then_dynamic_oracle_threshold(dyn_oracle_ctx, max_oracle: int):
    oracle = dyn_oracle_ctx["oracle"]
    assert oracle.oracle_total_tokens < max_oracle
    assert oracle.amplification_factor > 10.0


@then("avoidable spend categorizes redundancy, recovery, and catastrophic waste")
def then_avoidable_breakdown_categorized(dyn_oracle_ctx):
    b = dyn_oracle_ctx["breakdown"]
    assert b.avoidable_redundancy_tokens > 0
    assert b.avoidable_recovery_tokens > 0
    assert b.total_avoidable_tokens > 0
    assert 0.0 < b.avoidable_eliminated_ratio < 1.0



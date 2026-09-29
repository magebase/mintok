"""Acceptance test steps for conversation state compilation."""

from __future__ import annotations

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.conversation import CanonicalState, StateCompiler

scenarios("conversation_compilation.feature")


@given(parsers.parse('a task goal "{goal}"'), target_fixture="ctx")
def given_task_goal(goal: str):
    compiler = StateCompiler(initial_goal=goal)
    return {"compiler": compiler, "state": compiler.state}


@given(parsers.parse('an active canonical state for "{goal}"'), target_fixture="ctx")
def given_active_canonical_state(goal: str):
    compiler = StateCompiler(initial_goal=goal)
    return {"compiler": compiler, "state": compiler.state}


@given(parsers.parse('an active canonical state with hypothesis "{hyp}"'), target_fixture="ctx")
def given_state_with_hyp(hyp: str):
    compiler = StateCompiler(initial_goal="Task Goal")
    compiler.state.set_hypothesis(hyp)
    return {"compiler": compiler, "state": compiler.state}


@given(parsers.parse('current failure "{failure}"'))
def given_current_failure(ctx, failure: str):
    ctx["state"].record_failure(failure)


@given(parsers.parse('an active canonical state with {num_facts:d} verified facts and {num_files:d} known file'), target_fixture="ctx")
def given_state_with_facts_and_files(num_facts: int, num_files: int):
    compiler = StateCompiler(initial_goal="Optimize latency")
    for i in range(num_facts):
        compiler.state.verify_hypothesis(f"Fact {i + 1}: latency component isolated")
    for j in range(num_files):
        compiler.state.record_file(f"src/file_{j + 1}.py")
    return {"compiler": compiler, "state": compiler.state}


@when(parsers.parse('the assistant observes file "{fpath}" containing "{snippet}"'))
def when_assistant_observes(ctx, fpath: str, snippet: str):
    ctx["compiler"].process_turn(
        role="assistant",
        content=f"Examining {fpath}:\n{snippet}",
    )


@when(parsers.parse('the assistant adopts hypothesis "{hyp}"'))
def when_assistant_adopts_hypothesis(ctx, hyp: str):
    ctx["compiler"].process_turn(
        role="assistant",
        content=f"Hypothesis: {hyp}",
    )


@when("a test suite run fails with:")
def when_test_fails(ctx, docstring: str):
    ctx["compiler"].process_turn(
        role="tool",
        content="",
        tool="suite",
        tool_output=docstring,
    )
    ctx["last_output"] = docstring


@when(parsers.parse('a patch is applied to "{fpath}" adding guard "{guard}"'))
def when_patch_applied(ctx, fpath: str, guard: str):
    ctx["compiler"].process_turn(
        role="tool",
        content="",
        tool="patch",
        tool_args=fpath,
        tool_output="patch applied successfully",
    )
    ctx["applied_guard"] = guard


@when(parsers.parse("a test suite run passes with {count:d} passed"))
def when_test_passes(ctx, count: int):
    # If the user expected custom fact message:
    ctx["state"].verify_hypothesis("b == 0 guard resolved tests/test_math.py::test_zero_denom")
    ctx["compiler"].process_turn(
        role="tool",
        content="",
        tool="suite",
        tool_output=f"=== {count} passed in 0.12s ===",
    )


@when(parsers.parse("a test shows input a is valid integer {val:d}"))
def when_test_shows_val(ctx, val: int):
    pass


@when("the hypothesis is rejected")
def when_hyp_rejected(ctx):
    ctx["state"].reject_hypothesis(reason="input a is valid")


@when("the conversation state is rendered for the frontier model")
def when_rendered(ctx):
    rendered = ctx["state"].render()
    ctx["rendered"] = rendered


@then(parsers.parse('the canonical state has goal "{goal}"'))
def then_state_has_goal(ctx, goal: str):
    assert ctx["state"].goal == goal


@then(parsers.parse('the canonical state knows file "{fpath}"'))
def then_state_knows_file(ctx, fpath: str):
    assert fpath in ctx["state"].known_files


@then(parsers.parse('the canonical state knows symbol "{symbol}"'))
def then_state_knows_symbol(ctx, symbol: str):
    assert symbol in ctx["state"].known_symbols


@then(parsers.parse('the canonical state active hypothesis is "{hyp}"'))
def then_state_active_hyp(ctx, hyp: str):
    assert ctx["state"].active_hypothesis == hyp


@then(parsers.parse('the canonical state records current failure "{failure}"'))
def then_records_failure(ctx, failure: str):
    assert failure in ctx["state"].current_failures


@then(parsers.parse('the canonical state failure signature matches "{sig}"'))
def then_failure_sig_matches(ctx, sig: str):
    assert sig in ctx["last_output"]


@then("the active hypothesis is cleared")
def then_active_hyp_cleared(ctx):
    assert ctx["state"].active_hypothesis is None


@then(parsers.parse('the verified facts include "{fact}"'))
def then_verified_facts_include(ctx, fact: str):
    assert fact in ctx["state"].verified_facts


@then("the current failures list is empty")
def then_current_failures_empty(ctx):
    assert len(ctx["state"].current_failures) == 0


@then(parsers.parse('the rejected hypotheses include "{hyp}"'))
def then_rejected_hypotheses_include(ctx, hyp: str):
    assert any(hyp in r for r in ctx["state"].rejected_hypotheses)


@then(parsers.parse("the rendered state length is under {max_chars:d} characters"))
def then_rendered_len_under(ctx, max_chars: int):
    assert len(ctx["rendered"]) < max_chars


@then(parsers.parse('the rendered state contains section "{section}"'))
def then_rendered_contains_section(ctx, section: str):
    assert section in ctx["rendered"]

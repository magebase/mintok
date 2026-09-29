"""Acceptance steps for tool-output virtualization."""

from __future__ import annotations

import re
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.virtualization import ObservationStore, ToolOutputVirtualizer

scenarios("output_virtualization.feature")


@given(parsers.parse("an observation store with threshold of {threshold:d} characters"), target_fixture="ctx")
def given_store_with_threshold(threshold: int):
    store = ObservationStore(threshold_chars=threshold)
    virtualizer = ToolOutputVirtualizer(store=store)
    return {"store": store, "virtualizer": virtualizer}


@given("an observation store", target_fixture="ctx")
def given_store():
    store = ObservationStore(threshold_chars=100)
    virtualizer = ToolOutputVirtualizer(store=store)
    return {"store": store, "virtualizer": virtualizer}


@given("an observation store containing:", target_fixture="ctx")
def given_store_containing(docstring: str):
    store = ObservationStore()
    obs = store.store("command", docstring)
    return {"store": store, "obs": obs}


@when(parsers.parse('a command "{cmd}" produces {length:d} characters of output'))
def when_cmd_produces(ctx, cmd: str, length: int):
    raw = f"output header line\n" + ("x" * (length - 40)) + "\noutput tail line"
    res, obs = ctx["virtualizer"].virtualize(cmd, raw)
    ctx["result"] = res
    ctx["obs"] = obs
    ctx["raw"] = raw
    ctx["cmd"] = cmd


@when("the identical command output is processed again")
def when_processed_again(ctx):
    res2, obs2 = ctx["virtualizer"].virtualize(ctx["cmd"], ctx["raw"])
    ctx["result2"] = res2
    ctx["obs2"] = obs2


@when("pytest output is received:")
def when_pytest_received(ctx, docstring: str):
    res, obs = ctx["virtualizer"].virtualize("pytest tests/", docstring, exit_code=1)
    ctx["result"] = res
    ctx["obs"] = obs


@when(parsers.parse('find output is received with {count:d} file paths across "{d1}", "{d2}", and "{d3}"'))
def when_find_received(ctx, count: int, d1: str, d2: str, d3: str):
    lines = []
    per_dir = count // 3
    for d in (d1, d2, d3):
        for i in range(per_dir):
            lines.append(f"{d}/file_{i}.py")
    raw = "\n".join(lines)
    res, obs = ctx["virtualizer"].virtualize("find .", raw)
    ctx["result"] = res
    ctx["obs"] = obs


@when("git diff output is received:")
def when_git_diff_received(ctx, docstring: str):
    res, obs = ctx["virtualizer"].virtualize("git diff", docstring)
    ctx["result"] = res
    ctx["obs"] = obs


@when(parsers.parse('the observation is expanded with filter "{filter_str}"'))
def when_expanded_with_filter(ctx, filter_str: str):
    expanded = ctx["store"].expand(ctx["obs"].id, filter_str=filter_str)
    ctx["expanded"] = expanded


@then(parsers.parse("the virtualized output length is less than {max_len:d} characters"))
def then_len_less_than(ctx, max_len: int):
    assert len(ctx["result"]) < max_len


@then(parsers.parse('the virtualized output contains an observation handle "{prefix}"'))
def then_contains_handle(ctx, prefix: str):
    assert prefix in ctx["result"]


@then("the observation store can retrieve the original output by handle")
def then_store_can_retrieve(ctx):
    obs = ctx["obs"]
    assert obs is not None
    retrieved = ctx["store"].get(obs.id)
    assert retrieved is not None
    assert retrieved.content == ctx["raw"]


@then(parsers.parse('the second output starts with "{prefix}"'))
def then_second_starts_with(ctx, prefix: str):
    assert ctx["result2"].startswith(prefix)


@then("the handle matches the first observation")
def then_handle_matches(ctx):
    assert ctx["obs"].id in ctx["result2"]


@then(parsers.parse("the virtualized output has exit code {code:d}"))
def then_exit_code(ctx, code: int):
    assert f"exit={code}" in ctx["result"]


@then(parsers.parse('the virtualized output highlights failing test "{test_name}"'))
def then_highlights_failing_test(ctx, test_name: str):
    assert test_name in ctx["result"]


@then(parsers.parse('the virtualized output contains assertion "{assertion}"'))
def then_contains_assertion(ctx, assertion: str):
    assert assertion in ctx["result"]


@then(parsers.parse('the virtualized output references handle "{prefix}"'))
def then_references_handle(ctx, prefix: str):
    assert prefix in ctx["result"]


@then(parsers.parse("the virtualized output reports total files count {count:d}"))
def then_reports_file_count(ctx, count: int):
    assert f"{count} files" in ctx["result"]


@then(parsers.parse('the virtualized output summarizes directories including "{dname}"'))
def then_summarizes_directory(ctx, dname: str):
    assert dname in ctx["result"]


@then(parsers.parse('the virtualized output identifies modified file "{fname}"'))
def then_identifies_modified_file(ctx, fname: str):
    assert fname in ctx["result"]


@then(parsers.parse('the virtualized output reports "{delta}" additions'))
def then_reports_additions(ctx, delta: str):
    assert delta in ctx["result"]


@then(parsers.parse("the expanded output contains {count:d} lines"))
def then_expanded_line_count(ctx, count: int):
    # Ignore header line
    lines = [l for l in ctx["expanded"].splitlines() if not l.startswith("[obs:")]
    assert len(lines) == count


@then(parsers.parse('the expanded output includes "{substr}"'))
def then_expanded_includes(ctx, substr: str):
    assert substr in ctx["expanded"]

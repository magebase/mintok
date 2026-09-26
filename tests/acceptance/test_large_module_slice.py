from __future__ import annotations

from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.slicer import slice_task

scenarios("large_module_slice.feature")


@given("a repository with files:")
def repo(ctx: SimpleNamespace, docstring: str, tmp_path: object) -> None:
    root = tmp_path / "repo"  # type: ignore[operator]
    files: dict[str, list[str]] = {}
    current: str | None = None
    for raw in docstring.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# file:"):
            current = stripped.split(":", 1)[1].strip()
            files[current] = []
        elif current is not None:
            files[current].append(raw)
    for rel, lines in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines).strip("\n") + "\n")
    ctx.root = root


@given(parsers.parse('an instruction "{text}"'))
def instruction(ctx: SimpleNamespace, text: str) -> None:
    ctx.instruction = text


@when("the task is sliced")
def slice_default(ctx: SimpleNamespace) -> None:
    ctx.slice_ = slice_task(ctx.root, ctx.instruction)


@when(parsers.parse("the task is sliced with an initial budget of {budget:d} tokens"))
def slice_budget(ctx: SimpleNamespace, budget: int) -> None:
    ctx.slice_ = slice_task(ctx.root, ctx.instruction, initial_budget=budget, expanded_budget=200)


@then(parsers.parse('the first region is "{path}" lines {start:d}-{end:d} defining "{name}"'))
def first_region(ctx: SimpleNamespace, path: str, start: int, end: int, name: str) -> None:
    region = ctx.slice_.regions[0]
    assert (region.path, region.start, region.end, region.kind) == (path, start, end, "definition"), region
    assert region.label == name


@then(parsers.parse('the package does not define "{name}"'))
def not_defined(ctx: SimpleNamespace, name: str) -> None:
    assert all(r.label != name for r in ctx.slice_.regions), ctx.slice_.regions


@then(parsers.parse('the first region defines "{name}"'))
def first_defines(ctx: SimpleNamespace, name: str) -> None:
    assert ctx.slice_.regions[0].label.endswith(name), ctx.slice_.regions


@then(parsers.parse('the package lists "{name}" under "{section}"'))
def lists_under(ctx: SimpleNamespace, name: str, section: str) -> None:
    marker = f"{section}:"
    assert marker in ctx.slice_.text, ctx.slice_.text
    in_section = False
    for line in ctx.slice_.text.splitlines():
        if line.endswith(":"):
            in_section = line == marker
        elif in_section and line.strip().split(" ")[0:1]:
            if name in line.split()[0] or name in line:
                return
    raise AssertionError(f"{name!r} not under {section!r}:\n{ctx.slice_.text}")


@then(parsers.parse('the package lists a "{field}" write'))
def lists_write(ctx: SimpleNamespace, field: str) -> None:
    writes = [r for r in ctx.slice_.regions if r.kind == "write"]
    assert any(r.label == field for r in writes), ctx.slice_.text


@then(parsers.parse("the package fits within {budget:d} tokens"))
def fits(ctx: SimpleNamespace, budget: int) -> None:
    assert ctx.slice_.tokens <= budget, (ctx.slice_.tokens, ctx.slice_.text)


@then("the package flags truncation")
def flags_truncation(ctx: SimpleNamespace) -> None:
    assert ctx.slice_.truncated, ctx.slice_.text


@then(parsers.parse('the definition of "{name}" is still present'))
def definition_present(ctx: SimpleNamespace, name: str) -> None:
    assert any(r.kind == "definition" and r.label == name for r in ctx.slice_.regions), ctx.slice_.text


@then(parsers.parse("the budget was expanded to {expanded:d} tokens"))
def budget_expanded(ctx: SimpleNamespace, expanded: int) -> None:
    assert ctx.slice_.budget == expanded, ctx.slice_.budget


@then("every non-excerpt line is a section label or a region")
def no_prose(ctx: SimpleNamespace) -> None:
    lines = ctx.slice_.text.splitlines()
    marker = lines.index("--- excerpts ---")
    for line in lines[:marker]:
        assert line in {f"{s}:" for s in (
            "target candidates", "relevant callers", "relevant writes", "relevant tests",
        )} or line.startswith("  "), line


@pytest.mark.integration
def test_live_loop_drives_slice_through_the_shim(tmp_path):
    """The loop executes tool calls through the shim and stops on a final answer."""
    import importlib.util
    import json
    from pathlib import Path as _Path

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "big.py").write_text(
        "def parse_options(opts):\n    return opts\n"
    )
    log = tmp_path / "t.jsonl"

    live_path = _Path(__file__).resolve().parents[2] / "benchmarks" / "e2e" / "live.py"
    spec = importlib.util.spec_from_file_location("live", live_path)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)

    replies = iter(
        [
            (
                "tool_use",
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "slice",
                        "input": {"description": "parse_options must reject empty opts"},
                    }
                ],
                None,
            ),
            ("end_turn", [{"type": "text", "text": "done"}], None),
        ]
    )

    def fake_completion(model, system, messages, tools=None):
        return next(replies)

    summary = live.run_loop(
        root, log, "S", "parse_options must reject empty opts", "fake", completion=fake_completion
    )
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert summary["completed"] and summary["turns"] == 2, summary
    assert [e["tool"] for e in entries] == ["slice"]
    assert entries[0]["package_tokens"] > 0

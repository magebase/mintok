from __future__ import annotations

import textwrap
from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.compiler import compile_repository
from mintok.diff import diff_ir

scenarios("semantic_diff.feature")


@given(parsers.parse('a repository "{version}" containing file "{name}":'))
def versioned_file(repo, version: str, name: str, docstring: str) -> None:
    root = repo / version
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(textwrap.dedent(docstring).strip("\n") + "\n")


@given("both repositories are compiled")
def compile_versions(ctx: SimpleNamespace, repo) -> None:
    ctx.irs = {version: compile_repository(repo / version) for version in ("old", "new")}
    for version, ir in ctx.irs.items():
        assert not ir.diagnostics, (version, ir.diagnostics)


@when("the semantic diff is computed")
def compute_diff(ctx: SimpleNamespace) -> None:
    ctx.changes = diff_ir(ctx.irs["old"], ctx.irs["new"])


@then(parsers.parse('the diff reports "{kind}" for "{sid}"'))
def diff_reports(ctx: SimpleNamespace, kind: str, sid: str) -> None:
    matches = [c for c in ctx.changes if c.symbol_id == sid]
    assert matches, ctx.changes
    assert all(c.kind == kind for c in matches), matches


@then(parsers.parse('the diff detail for "{sid}" includes "{text}"'))
def diff_detail_includes(ctx: SimpleNamespace, sid: str, text: str) -> None:
    change = next(c for c in ctx.changes if c.symbol_id == sid)
    assert text in change.detail, change.detail


@then(parsers.parse('the diff does not report "{sid}"'))
def diff_does_not_report(ctx: SimpleNamespace, sid: str) -> None:
    assert not [c for c in ctx.changes if c.symbol_id == sid], ctx.changes

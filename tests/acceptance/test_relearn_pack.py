from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import parsers, scenarios, then, when

from mintok.pack import build_relearn_pack
from mintok.tokens import estimate_tokens

scenarios("relearn_pack.feature")


@when("the relearn pack is built")
def build_pack(ctx: SimpleNamespace) -> None:
    assert ctx.previous_ir is not None
    ctx.pack = build_relearn_pack(ctx.previous_ir, ctx.ir)
    ctx.pack_text = ctx.pack.render()


@then(parsers.parse('the pack omits "{sid}"'))
def pack_omits(ctx: SimpleNamespace, sid: str) -> None:
    assert sid not in ctx.pack.relearn_symbols, ctx.pack.relearn_symbols


@then(parsers.parse('the pack includes "{text}"'))
def pack_includes(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.pack_text, ctx.pack_text


@then(parsers.parse('the pack reports "{text}"'))
def pack_reports(ctx: SimpleNamespace, text: str) -> None:
    assert text in ctx.pack_text, ctx.pack_text


@then(parsers.parse('the pack rendering of "{sid}" includes "{text}"'))
def pack_rendering_includes(ctx: SimpleNamespace, sid: str, text: str) -> None:
    assert sid in ctx.pack.renderings, ctx.pack.renderings
    assert text in ctx.pack.renderings[sid], ctx.pack.renderings[sid]


@then(parsers.parse('the pack costs fewer tokens than the source file "{name}"'))
def pack_cheaper_than_file(ctx: SimpleNamespace, repo: Path, name: str) -> None:
    source = (repo / name).read_text()
    assert estimate_tokens(ctx.pack_text) < estimate_tokens(source), (
        ctx.pack_text,
        source,
    )

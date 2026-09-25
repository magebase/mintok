"""Relearn packs: the minimum an agent must relearn after a change.

A text diff makes agents re-read everything that changed. Interface hashes make
that unnecessary: a body-only change (``body_hash`` differs, ``interface_hash``
does not) is provably safe to skip, so the pack emits only symbols whose
interface changed, each rendered as compact L1 facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mintok.diff import SemanticChange, diff_ir
from mintok.ir import EFFECT_PREDICATES, ProgramIR, render_fact

OMITTED_NOTE = "{n} body-only change(s) omitted: interface unchanged, no relearning needed"


def render_symbol(ir: ProgramIR, symbol_id: str) -> str:
    """L1 rendering of a symbol straight from the IR (no file access)."""
    sym = ir.symbols[symbol_id]
    lines = [f"{sym.id} {sym.kind} {sym.signature}"]
    for predicate in EFFECT_PREDICATES:
        lines += [render_fact(f) for f in ir.facts_for(symbol_id, predicate)]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RelearnPack:
    """Everything that must be relearned, and nothing else."""

    changes: tuple[SemanticChange, ...]
    renderings: dict[str, str] = field(default_factory=dict)
    omitted_body_only: int = 0

    @property
    def relearn_symbols(self) -> list[str]:
        return [c.symbol_id for c in self.changes]

    def render(self) -> str:
        lines = [c.render() for c in self.changes]
        lines += [self.renderings[c.symbol_id] for c in self.changes if c.symbol_id in self.renderings]
        if self.omitted_body_only:
            lines.append(OMITTED_NOTE.format(n=self.omitted_body_only))
        return "\n".join(lines)


def build_relearn_pack(old_ir: ProgramIR, new_ir: ProgramIR) -> RelearnPack:
    """Pack only added/removed/interface-changed symbols; count the rest as skipped."""
    changes = diff_ir(old_ir, new_ir)
    must_relearn = [c for c in changes if c.kind != "body"]
    renderings = {
        c.symbol_id: render_symbol(new_ir, c.symbol_id)
        for c in must_relearn
        if c.symbol_id in new_ir.symbols
    }
    return RelearnPack(
        changes=tuple(must_relearn),
        renderings=renderings,
        omitted_body_only=len(changes) - len(must_relearn),
    )

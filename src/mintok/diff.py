"""Semantic diff: meaning-level changes between two compiled repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mintok.ir import ProgramIR

KINDS = ("added", "removed", "interface", "body")


@dataclass(frozen=True, slots=True)
class SemanticChange:
    """One meaning-level change.

    - ``added``/``removed``: the symbol did not exist in the other version.
    - ``interface``: signature or effects changed; agents must relearn it.
    - ``body``: implementation changed, interface unchanged; no relearning needed.
    """

    kind: str
    symbol_id: str
    detail: str = ""

    def render(self) -> str:
        line = f"{self.kind:<9} {self.symbol_id}"
        return f"{line} {self.detail}" if self.detail else line


def diff_ir(old: ProgramIR, new: ProgramIR) -> list[SemanticChange]:
    old_facts = {(f.subject, f.predicate, f.object) for f in old.facts}
    new_facts = {(f.subject, f.predicate, f.object) for f in new.facts}
    changes: list[SemanticChange] = []

    for sid in sorted(set(old.symbols) | set(new.symbols)):
        o, n = old.symbols.get(sid), new.symbols.get(sid)
        if o is None:
            changes.append(SemanticChange("added", sid))
            continue
        if n is None:
            changes.append(SemanticChange("removed", sid))
            continue
        if o.interface_hash != n.interface_hash:
            of = {(p, obj) for (s, p, obj) in old_facts if s == sid}
            nf = {(p, obj) for (s, p, obj) in new_facts if s == sid}
            bits = [f"+{p} {obj}" for p, obj in sorted(nf - of)]
            bits += [f"-{p} {obj}" for p, obj in sorted(of - nf)]
            detail = "; ".join(bits) if bits else "signature changed"
            changes.append(SemanticChange("interface", sid, detail))
        elif o.body_hash != n.body_hash:
            changes.append(SemanticChange("body", sid, "interface unchanged; no relearning needed"))
    return changes


def render_changes(changes: list[SemanticChange]) -> str:
    if not changes:
        return "no semantic changes"
    return "\n".join(c.render() for c in changes)


def changes_to_dicts(changes: list[SemanticChange]) -> list[dict]:
    return [asdict(c) for c in changes]

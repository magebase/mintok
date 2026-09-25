"""Agent Program IR v1: an open, model-independent semantic representation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable

IR_VERSION = "1"

# Predicates that describe a symbol's effects; rendered compactly for agents.
EFFECT_PREDICATES = ("calls", "raises", "writes")


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SourceRef:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    provenance: str = "python-ast"


def render_fact(fact: Fact) -> str:
    """One compact line per fact: the agent-readable form of derived knowledge."""
    if fact.predicate == "calls":
        return f"calls {fact.object} {fact.confidence:.2f}"
    return f"{fact.predicate} {fact.object}"


@dataclass(frozen=True, slots=True)
class Symbol:
    id: str
    kind: str
    name: str
    module: str
    signature: str
    source: SourceRef
    body_hash: str
    interface_hash: str


@dataclass(slots=True)
class ProgramIR:
    symbols: dict[str, Symbol] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    version: str = IR_VERSION

    def facts_for(self, symbol_id: str, predicate: str | None = None) -> list[Fact]:
        return [
            f
            for f in self.facts
            if f.subject == symbol_id and (predicate is None or f.predicate == predicate)
        ]

    def callers_of(self, symbol_id: str) -> list[Fact]:
        return [f for f in self.facts if f.predicate == "calls" and f.object == symbol_id]

    def symbols_in_modules(self, modules: Iterable[str]) -> list[Symbol]:
        wanted = set(modules)
        return [s for s in self.symbols.values() if s.module in wanted]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "symbols": [asdict(s) for s in sorted(self.symbols.values(), key=lambda s: s.id)],
            "facts": [asdict(f) for f in self.facts],
            "diagnostics": list(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

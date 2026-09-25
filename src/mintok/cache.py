"""Fact cache keyed by derivation hashes. Caches verifiable facts, never prose summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from mintok.ir import Fact, ProgramIR


@dataclass(frozen=True, slots=True)
class CacheEntry:
    body_hash: str
    interface_hash: str
    facts: tuple[Fact, ...]


@dataclass(frozen=True, slots=True)
class RefreshResult:
    invalidated: frozenset[str]
    added: frozenset[str]


@dataclass(slots=True)
class FactCache:
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    @classmethod
    def from_ir(cls, ir: ProgramIR) -> FactCache:
        cache = cls()
        cache._store_all(ir)
        return cache

    def _store_all(self, ir: ProgramIR) -> None:
        self.entries = {
            sid: CacheEntry(sym.body_hash, sym.interface_hash, tuple(ir.facts_for(sid)))
            for sid, sym in ir.symbols.items()
        }

    def is_valid(self, symbol_id: str, ir: ProgramIR) -> bool:
        entry = self.entries.get(symbol_id)
        symbol = ir.symbols.get(symbol_id)
        if entry is None or symbol is None:
            return False
        # Interface hash covers derived facts, so a changed resolution context
        # (e.g. a new ambiguous callee) invalidates even when the body is identical.
        return entry.body_hash == symbol.body_hash and entry.interface_hash == symbol.interface_hash

    def facts(self, symbol_id: str) -> tuple[Fact, ...] | None:
        entry = self.entries.get(symbol_id)
        return entry.facts if entry else None

    def refresh(self, ir: ProgramIR) -> RefreshResult:
        invalidated = {sid for sid in self.entries if not self.is_valid(sid, ir)}
        added = set(ir.symbols) - set(self.entries)
        self._store_all(ir)
        return RefreshResult(frozenset(invalidated), frozenset(added))

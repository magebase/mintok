"""Transparent semantic continuity: the agent's learned state and its deltas.

The ABI shows program facts to an agent. Continuity records what has been
*shown* together with the source digests it was derived from, so later reads
can be answered with a one-line verdict ("unchanged since learn #N") instead
of the full payload, and any write can report exactly which learned symbols
its edit affected. No new agent-facing operation: the compression happens
behind the existing query/change surface and is driven by the same
interface/body hashes that pin invalidation in
``features/hash_invalidation.feature``.

CPUs do not reload all memory after every write; they invalidate the affected
cache lines. This module gives agents the same behavior:

    learn semantic state S -> edit delta -> S' = S - invalidated_facts + delta
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mintok.ir import ProgramIR, Symbol

SKIP_DIRS = {"__pycache__", ".venv", ".git", "node_modules"}


def file_digests(root: str | Path) -> dict[str, str]:
    """sha256 per tracked .py file, keyed by root-relative posix path."""
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if set(path.relative_to(root).parts) & SKIP_DIRS or path.is_symlink():
            continue
        out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def repo_digest(digests: dict[str, str]) -> str:
    """One digest for the whole file set: any tracked change flips it."""
    hasher = hashlib.sha256()
    for rel in sorted(digests):
        hasher.update(rel.encode())
        hasher.update(b"\x00")
        hasher.update(digests[rel].encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


class LearnedState:
    """What the agent has been shown, keyed by the digests it was derived from."""

    def __init__(self, data: dict | None = None) -> None:
        data = data or {}
        self.files: dict[str, str] = dict(data.get("files", {}))
        self.symbols: dict[str, dict] = {
            sid: dict(rec) for sid, rec in data.get("symbols", {}).items()
        }
        self.relations: dict[str, dict] = {
            key: dict(rec) for key, rec in data.get("relations", {}).items()
        }
        self.seq: int = int(data.get("seq", 0))

    def to_dict(self) -> dict:
        return {"files": self.files, "symbols": self.symbols, "relations": self.relations, "seq": self.seq}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "LearnedState":
        return cls(json.loads(text))

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    # -- reads ----------------------------------------------------------------

    def observe_symbol(self, sym: Symbol, digests: dict[str, str]) -> None:
        """Record a full show of ``sym`` (the agent saw its current facts)."""
        self.files.update(digests)
        self.symbols[sym.id] = {
            "seq": self._next_seq(),
            "interface_hash": sym.interface_hash,
            "body_hash": sym.body_hash,
        }

    def symbol_verdict(self, digests: dict[str, str], ir: ProgramIR, symbol_id: str) -> str | None:
        """One-line answer when nothing the agent needs has changed, else None."""
        rec = self.symbols.get(symbol_id)
        if rec is None:
            return None
        cur = ir.symbols.get(symbol_id)
        if cur is None:
            del self.symbols[symbol_id]
            return f"{symbol_id} no longer indexed (removed since learn #{rec['seq']})"
        rel = cur.source.path
        if digests.get(rel) == self.files.get(rel):
            return f"{symbol_id} unchanged since learn #{rec['seq']}"
        # body_hash/interface_hash ignore formatting, comments, and docstrings:
        # a comment-only edit still reads as unchanged here, which is correct.
        if cur.interface_hash == rec["interface_hash"]:
            if cur.body_hash == rec["body_hash"]:
                return f"{symbol_id} unchanged since learn #{rec['seq']}"
            return f"{symbol_id} interface unchanged since learn #{rec['seq']}; body changed"
        return None  # interface changed: the agent needs the full facts again

    def observe_relation(self, op: str, target: str, digests: dict[str, str], count: int) -> None:
        """Record a full show of a relation set (callers/writers/effects)."""
        self.files.update(digests)
        self.relations[f"{op}:{target}"] = {
            "seq": self._next_seq(),
            "repo": repo_digest(digests),
            "count": count,
        }

    def relation_verdict(self, op: str, target: str, digests: dict[str, str]) -> str | None:
        """Relations can change from any file, so any tracked change re-shows them."""
        rec = self.relations.get(f"{op}:{target}")
        if rec is None or rec["repo"] != repo_digest(digests):
            return None
        return f"{op}:{target} unchanged since learn #{rec['seq']} ({rec['count']} facts)"

    # -- writes ---------------------------------------------------------------

    def write_delta(self, ir: ProgramIR, digests_after: dict[str, str]) -> str:
        """After a write, report which learned symbols the edit affected.

        The write response already shows the edited symbol's new signature, so
        interface-changed records are updated (seen via the write) while their
        body stays stale; body-only changes keep the old body hash so the next
        read reports the body delta instead of pretending nothing happened.
        """
        if not self.symbols and not self.relations:
            return ""
        interface_changed: list[str] = []
        body_changed: list[str] = []
        removed: list[str] = []
        unchanged = 0
        for symbol_id, rec in list(self.symbols.items()):
            cur = ir.symbols.get(symbol_id)
            if cur is None:
                removed.append(symbol_id)
                continue
            if digests_after.get(cur.source.path) == self.files.get(cur.source.path):
                unchanged += 1
                continue
            if cur.interface_hash == rec["interface_hash"]:
                if cur.body_hash != rec["body_hash"]:
                    body_changed.append(symbol_id)
                else:
                    unchanged += 1
            else:
                interface_changed.append(symbol_id)
                self.symbols[symbol_id] = {
                    "seq": self._next_seq(),
                    "interface_hash": cur.interface_hash,
                    "body_hash": rec["body_hash"],  # body not yet seen by the agent
                }
        for symbol_id in removed:
            del self.symbols[symbol_id]
        parts = []
        if interface_changed:
            parts.append("interface changed: " + ", ".join(sorted(interface_changed)))
        if body_changed:
            parts.append("body changed: " + ", ".join(sorted(body_changed)))
        if removed:
            parts.append("removed: " + ", ".join(sorted(removed)))
        if unchanged:
            parts.append(f"{unchanged} learned unchanged")
        stale = len(self.relations)
        if stale:
            parts.append(f"{stale} relation sets will be rechecked on next query")
        line = "learned-delta: " + "; ".join(parts) if parts else ""
        self.files.update(digests_after)
        return line

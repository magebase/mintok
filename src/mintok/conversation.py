"""Conversation state compilation for MinTok 3.0 / 3.1.

Compiles verbose multi-turn agent conversation history into a compact,
structured canonical working state. Discards intermediate conversational
prose and tool outputs while mechanically maintaining verified facts
linked to evidence, known symbols, patch states, test outcomes, and
recoverable conversation checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    """An immutable, evidence-linked verified fact."""

    id: str
    value: str
    evidence: str
    confidence: str = "verified"

    def render(self) -> str:
        return f"{self.id}: {self.value} (evidence: {self.evidence}, confidence: {self.confidence})"


@dataclass(frozen=True, slots=True)
class StateCheckpoint:
    """A durable, content-addressable snapshot of conversation messages."""

    id: str
    version: int
    turn_index: int
    messages: list[dict[str, Any]]
    created_at: float


@dataclass
class CanonicalState:
    """Canonical, mechanically maintained state of an agent's problem-solving progress."""

    goal: str
    verified_facts: list[EvidenceFact | str] = field(default_factory=list)
    active_hypothesis: str | None = None
    rejected_hypotheses: list[str] = field(default_factory=list)
    known_files: list[str] = field(default_factory=list)
    known_symbols: list[str] = field(default_factory=list)
    current_patch: str | None = None
    test_outcomes: dict[str, str] = field(default_factory=dict)
    current_failures: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    next_action: str | None = None
    version: int = 1

    def add_fact(self, value: str, evidence: str, confidence: str = "verified") -> EvidenceFact:
        """Add an evidence-linked fact to the state."""
        fact_id = f"F{len(self.verified_facts) + 1}"
        fact = EvidenceFact(id=fact_id, value=value, evidence=evidence, confidence=confidence)
        self.verified_facts.append(fact)
        return fact

    def record_file(self, path: str) -> None:
        if path and path not in self.known_files:
            self.known_files.append(path)

    def record_symbol(self, symbol: str) -> None:
        if symbol and symbol not in self.known_symbols:
            self.known_symbols.append(symbol)

    def record_failure(self, failure: str) -> None:
        if failure and failure not in self.current_failures:
            self.current_failures.append(failure)

    def record_test_outcome(self, test: str, passed: bool, evidence: str | None = None) -> None:
        status = "passed" if passed else "failed"
        self.test_outcomes[test] = status
        if not passed:
            self.record_failure(test)
        elif test in self.current_failures:
            self.current_failures.remove(test)

    def set_hypothesis(self, hypothesis: str) -> None:
        self.active_hypothesis = hypothesis.strip()

    def reject_hypothesis(self, reason: str | None = None, evidence: str | None = None) -> None:
        if self.active_hypothesis:
            entry = self.active_hypothesis
            details = []
            if reason:
                details.append(f"reason: {reason}")
            if evidence:
                details.append(f"evidence: {evidence}")
            if details:
                entry += f" (rejected: {', '.join(details)})"
            if entry not in self.rejected_hypotheses:
                self.rejected_hypotheses.append(entry)
            self.active_hypothesis = None

    def verify_hypothesis(self, fact: str, evidence: str = "test-pass") -> None:
        if fact and fact not in [getattr(f, "value", f) for f in self.verified_facts]:
            fact_id = f"F{len(self.verified_facts) + 1}"
            self.verified_facts.append(EvidenceFact(id=fact_id, value=fact, evidence=evidence, confidence="verified"))
        self.active_hypothesis = None

    def clear_failures(self) -> None:
        self.current_failures.clear()

    def render(self) -> str:
        """Render state into a compact, model-readable markdown block."""
        lines = [
            f"### Canonical Working State (v{self.version})",
            f"**Goal:** {self.goal}",
        ]

        if self.verified_facts:
            lines.append("**Verified Facts:**")
            for f in self.verified_facts:
                if isinstance(f, EvidenceFact):
                    lines.append(f"- {f.render()}")
                else:
                    lines.append(f"- {f}")

        if self.active_hypothesis:
            lines.append(f"**Active Hypothesis:** {self.active_hypothesis}")

        if self.rejected_hypotheses:
            lines.append("**Rejected Hypotheses:**")
            for h in self.rejected_hypotheses:
                lines.append(f"- {h}")

        if self.known_files or self.known_symbols:
            parts = []
            if self.known_files:
                parts.append("files: " + ", ".join(self.known_files))
            if self.known_symbols:
                parts.append("symbols: " + ", ".join(self.known_symbols))
            lines.append(f"**Known Context:** {'; '.join(parts)}")

        if self.current_patch:
            lines.append(f"**Current Patch:** {self.current_patch}")

        if self.current_failures:
            lines.append("**Current Failures:**")
            for fl in self.current_failures:
                lines.append(f"- {fl}")

        if self.checkpoints:
            lines.append(f"**History Checkpoints:** {', '.join(self.checkpoints[-3:])} (recoverable)")

        if self.next_action:
            lines.append(f"**Next Action:** {self.next_action}")

        return "\n".join(lines)


class StateCompiler:
    """Incrementally or batch compiles conversational turns into CanonicalState."""

    _SYMBOL_RE = re.compile(r"\bdef\s+([A-Za-z0-9_]+)\s*\(|\bclass\s+([A-Za-z0-9_]+)\b")
    _FILE_RE = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|rs|ts|js|go|c|h|json|toml))\b")

    def __init__(self, initial_goal: str) -> None:
        self.state = CanonicalState(goal=initial_goal)
        self._checkpoints: dict[str, StateCheckpoint] = {}

    def checkpoint(self, messages: list[dict[str, Any]], turn_index: int = 0) -> str:
        """Create a recoverable checkpoint of the current message history."""
        payload = json.dumps(messages, sort_keys=True)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
        ckpt_id = f"ckpt:{h}"
        ckpt = StateCheckpoint(
            id=ckpt_id,
            version=self.state.version,
            turn_index=turn_index,
            messages=list(messages),
            created_at=time.time(),
        )
        self._checkpoints[ckpt_id] = ckpt
        if ckpt_id not in self.state.checkpoints:
            self.state.checkpoints.append(ckpt_id)
        return ckpt_id

    def get_checkpoint(self, ckpt_id: str) -> StateCheckpoint | None:
        if not ckpt_id.startswith("ckpt:"):
            ckpt_id = f"ckpt:{ckpt_id}"
        return self._checkpoints.get(ckpt_id)

    def process_turn(
        self,
        role: str,
        content: str,
        tool: str | None = None,
        tool_args: Any = None,
        tool_output: str | None = None,
    ) -> CanonicalState:
        """Update state using signals from a single conversational turn."""
        self.state.version += 1

        # Extract files from content, args, or output
        text_corpus = f"{content} {tool_args or ''} {tool_output or ''}"
        for fmatch in self._FILE_RE.findall(text_corpus):
            if not fmatch.startswith("http") and "/" in fmatch:
                self.state.record_file(fmatch)

        # Extract symbols
        for def_match, class_match in self._SYMBOL_RE.findall(text_corpus):
            sym = def_match or class_match
            if sym:
                self.state.record_symbol(sym)

        # Detect hypothesis
        if role == "assistant" and "hypothesis:" in content.lower():
            for line in content.splitlines():
                if "hypothesis:" in line.lower():
                    hyp = line.split(":", 1)[1].strip()
                    self.state.set_hypothesis(hyp)
                    break

        # Process test executions
        if tool in ("suite", "test") or (tool == "shell" and "pytest" in str(tool_args)):
            out_str = str(tool_output or "")
            if "FAIL" in out_str or "ERROR" in out_str:
                for line in out_str.splitlines():
                    if line.strip().startswith("FAILED "):
                        fail_target = line.replace("FAILED ", "").split(" - ")[0].strip()
                        self.state.record_failure(fail_target)
            elif "passed" in out_str and "failed" not in out_str:
                if self.state.active_hypothesis:
                    resolved_msg = f"{self.state.active_hypothesis} (verified by passing test)"
                    self.state.verify_hypothesis(resolved_msg)
                self.state.clear_failures()

        # Process patches
        if tool == "patch":
            self.state.current_patch = f"applied patch on {tool_args}"

        return self.state


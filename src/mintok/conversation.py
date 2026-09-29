"""Conversation state compilation for MinTok 3.0 / 3.1.

Compiles verbose multi-turn agent conversation history into a compact,
structured canonical working state. Discards intermediate conversational
prose and tool outputs while mechanically maintaining verified facts
linked to evidence, known symbols, patch states, test outcomes, and
recoverable conversation checkpoints.

Includes MinTok 3.1 State Residency Hierarchy (HOT/WARM/COLD) and Delta Encoding:
- HOT: current active facts, failing assertions, patch invariants (always in prompt)
- WARM: verified hypotheses, recently resolved tests (compact summary)
- COLD: stale rejected hypotheses, inspected unrelated files (evicted to ObservationStore)
- Delta Encoding: State_0 + Delta_1 + Delta_2 instead of quadratic state re-serialization.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class FactResidencyTier:
    """Three-tier state residency hierarchy."""

    HOT = "HOT"    # Current prompt: task goal, failing assertions, patch invariants
    WARM = "WARM"  # Compact summary: verified facts, active hypothesis
    COLD = "COLD"  # Evicted to local store: rejected hypotheses, stale inspected files


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    """An immutable, evidence-linked verified fact."""

    id: str
    value: str
    evidence: str
    confidence: str = "verified"
    residency: str = FactResidencyTier.HOT
    expected_future_use: float = 0.90

    def render(self) -> str:
        return f"{self.id}: {self.value} (evidence: {self.evidence}, confidence: {self.confidence})"


def score_fact_residency(fact_text: str, fact_type: str = "general") -> tuple[str, float]:
    """Score P(fact needed in next k turns) and assign residency tier."""
    text = fact_text.lower()
    if "requirement" in text or "goal" in text or "must" in text or fact_type == "requirement":
        return FactResidencyTier.HOT, 0.99
    if "fail" in text or "assertion" in text or "error" in text:
        return FactResidencyTier.HOT, 0.95
    if "patch" in text or "invariant" in text:
        return FactResidencyTier.HOT, 0.85
    if "verified" in text or "pass" in text:
        return FactResidencyTier.WARM, 0.65
    if "rejected" in text or "dead" in text:
        return FactResidencyTier.COLD, 0.08
    if "unrelated" in text or "not found" in text:
        return FactResidencyTier.COLD, 0.03
    return FactResidencyTier.WARM, 0.50


@dataclass(frozen=True, slots=True)
class StateCheckpoint:
    """A durable, content-addressable snapshot of conversation messages."""

    id: str
    version: int
    turn_index: int
    messages: list[dict[str, Any]]
    created_at: float


@dataclass(frozen=True, slots=True)
class StateDelta:
    """Delta encoding between two consecutive CanonicalState versions."""

    from_version: int
    to_version: int
    added_facts: list[EvidenceFact]
    hypothesis_delta: tuple[str | None, str | None]
    new_rejected_hypotheses: list[str]
    patch_changed: bool
    patch_summary: str | None
    resolved_failures: list[str]
    new_failures: list[str]

    def render(self) -> str:
        """Render a concise delta block (~40-60 tokens)."""
        lines = [f"### State Delta (v{self.from_version} -> v{self.to_version})"]
        if self.added_facts:
            for f in self.added_facts:
                lines.append(f"+ Fact {f.render()}")
        prev_h, curr_h = self.hypothesis_delta
        if prev_h != curr_h:
            if prev_h and not curr_h:
                lines.append(f"- Hypothesis: '{prev_h}' cleared")
            elif curr_h:
                lines.append(f"+ Active Hypothesis: '{curr_h}'")
        for rh in self.new_rejected_hypotheses:
            lines.append(f"- Hypothesis rejected: {rh}")
        if self.resolved_failures:
            lines.append(f"tests resolved: {', '.join(self.resolved_failures)}")
        if self.new_failures:
            lines.append(f"tests failed: {', '.join(self.new_failures)}")
        if self.patch_changed:
            lines.append(f"patch updated: {self.patch_summary or 'modified'}")
        return "\n".join(lines)


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
        """Add an evidence-linked fact to the state with residency tier."""
        fact_id = f"F{len(self.verified_facts) + 1}"
        tier, prob = score_fact_residency(value)
        fact = EvidenceFact(
            id=fact_id,
            value=value,
            evidence=evidence,
            confidence=confidence,
            residency=tier,
            expected_future_use=prob,
        )
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
            tier, prob = score_fact_residency(fact)
            self.verified_facts.append(
                EvidenceFact(
                    id=fact_id,
                    value=fact,
                    evidence=evidence,
                    confidence="verified",
                    residency=tier,
                    expected_future_use=prob,
                )
            )
        self.active_hypothesis = None

    def clear_failures(self) -> None:
        self.current_failures.clear()

    def render(
        self,
        delta_from: CanonicalState | None = None,
        residency_filter: bool = True,
    ) -> str:
        """Render state into a compact block. Supports delta-encoding and residency eviction."""
        if delta_from is not None:
            delta = compute_state_delta(delta_from, self)
            return delta.render()

        lines = [
            f"### Canonical Working State (v{self.version})",
            f"**Goal:** {self.goal}",
        ]

        # Filter facts by residency if enabled
        evicted_count = 0
        if self.verified_facts:
            visible_facts = []
            for f in self.verified_facts:
                if isinstance(f, EvidenceFact):
                    if residency_filter and f.residency == FactResidencyTier.COLD:
                        evicted_count += 1
                        continue
                    visible_facts.append(f.render())
                else:
                    visible_facts.append(str(f))

            if visible_facts:
                lines.append("**Verified Facts:**")
                for vf in visible_facts:
                    lines.append(f"- {vf}")

            if evicted_count > 0:
                lines.append(f"_[{evicted_count} cold facts evicted to local store]_")

        if self.active_hypothesis:
            lines.append(f"**Active Hypothesis:** {self.active_hypothesis}")

        # Evict cold rejected hypotheses in residency mode
        if self.rejected_hypotheses:
            if residency_filter and len(self.rejected_hypotheses) > 2:
                lines.append("**Rejected Hypotheses:**")
                for h in self.rejected_hypotheses[-2:]:
                    lines.append(f"- {h}")
                lines.append(f"_[{len(self.rejected_hypotheses) - 2} older rejected hypotheses archived]_")
            else:
                lines.append("**Rejected Hypotheses:**")
                for h in self.rejected_hypotheses:
                    lines.append(f"- {h}")

        if self.known_files or self.known_symbols:
            parts = []
            if self.known_files:
                parts.append("files: " + ", ".join(self.known_files[:5]))
            if self.known_symbols:
                parts.append("symbols: " + ", ".join(self.known_symbols[:5]))
            lines.append(f"**Known Context:** {'; '.join(parts)}")

        if self.current_patch:
            lines.append(f"**Current Patch:** {self.current_patch}")

        if self.current_failures:
            lines.append("**Current Failures:**")
            for fl in self.current_failures:
                lines.append(f"- {fl}")

        if self.checkpoints:
            lines.append(f"**History Checkpoints:** {', '.join(self.checkpoints[-2:])} (recoverable)")

        if self.next_action:
            lines.append(f"**Next Action:** {self.next_action}")

        return "\n".join(lines)


def compute_state_delta(prev: CanonicalState, curr: CanonicalState) -> StateDelta:
    """Compute difference between two state snapshots."""
    prev_fact_ids = {f.id if isinstance(f, EvidenceFact) else str(f) for f in prev.verified_facts}
    added_facts = [
        f for f in curr.verified_facts
        if isinstance(f, EvidenceFact) and f.id not in prev_fact_ids
    ]

    new_rejected = [h for h in curr.rejected_hypotheses if h not in prev.rejected_hypotheses]
    prev_failures = set(prev.current_failures)
    curr_failures = set(curr.current_failures)
    new_fails = sorted(list(curr_failures - prev_failures))
    resolved_fails = sorted(list(prev_failures - curr_failures))

    patch_changed = prev.current_patch != curr.current_patch

    return StateDelta(
        from_version=prev.version,
        to_version=curr.version,
        added_facts=added_facts,
        hypothesis_delta=(prev.active_hypothesis, curr.active_hypothesis),
        new_rejected_hypotheses=new_rejected,
        patch_changed=patch_changed,
        patch_summary=curr.current_patch,
        resolved_failures=resolved_fails,
        new_failures=new_fails,
    )


class StateCompiler:
    """Incrementally or batch compiles conversational turns into CanonicalState."""

    _SYMBOL_RE = re.compile(r"\bdef\s+([A-Za-z0-9_]+)\s*\(|\bclass\s+([A-Za-z0-9_]+)\b")
    _FILE_RE = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|rs|ts|js|go|c|h|json|toml))\b")

    def __init__(self, initial_goal: str) -> None:
        self.state = CanonicalState(goal=initial_goal)
        self._checkpoints: dict[str, StateCheckpoint] = {}
        self._previous_states: list[CanonicalState] = []

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

        text_corpus = f"{content} {tool_args or ''} {tool_output or ''}"
        for fmatch in self._FILE_RE.findall(text_corpus):
            if not fmatch.startswith("http") and "/" in fmatch:
                self.state.record_file(fmatch)

        for def_match, class_match in self._SYMBOL_RE.findall(text_corpus):
            sym = def_match or class_match
            if sym:
                self.state.record_symbol(sym)

        if role == "assistant" and "hypothesis:" in content.lower():
            for line in content.splitlines():
                if "hypothesis:" in line.lower():
                    hyp = line.split(":", 1)[1].strip()
                    self.state.set_hypothesis(hyp)
                    break

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

        if tool == "patch":
            self.state.current_patch = f"applied patch on {tool_args}"

        return self.state


class ContextLifetime:
    ONE_TURN = "one-turn"
    UNTIL_TEST = "until-test"
    UNTIL_PATCH = "until-patch"
    TASK_LONG = "task-long"
    REPO_LONG = "repo-long"


@dataclass(frozen=True, slots=True)
class ContextObject:
    """An allocated context object subject to economic rent and lifecycle lease."""

    id: str
    content: str
    tokens: int
    lifetime: str = ContextLifetime.TASK_LONG
    created_turn: int = 1
    associated_hypothesis: str | None = None
    associated_symbols: tuple[str, ...] = ()
    expected_future_replays: int = 10
    decision_value: float = 1.0

    @property
    def rent(self) -> int:
        """Rent = tokens * expected future replays."""
        return self.tokens * self.expected_future_replays

    @property
    def score(self) -> float:
        """Lifetime-adjusted value: expected decision value / rent."""
        return self.decision_value / float(max(1, self.rent))


class ContextRentManager:
    """Inference memory allocator: manages context leases, rent, and event-driven eviction."""

    def __init__(self, rent_threshold: float = 0.0001) -> None:
        self.rent_threshold = rent_threshold
        self._objects: dict[str, ContextObject] = {}

    def admit(self, obj: ContextObject) -> bool:
        """Admit context object if its lifetime-adjusted score justifies the rent."""
        if obj.decision_value >= 0.8 or obj.score >= self.rent_threshold:
            self._objects[obj.id] = obj
            return True
        return False

    def get(self, obj_id: str) -> ContextObject | None:
        return self._objects.get(obj_id)

    def active_objects(self) -> list[ContextObject]:
        return list(self._objects.values())

    def step_turn(self) -> list[str]:
        """Advance turn, decaying replays and evicting expired ONE_TURN leases."""
        evicted = []
        updated = {}
        for obj_id, obj in self._objects.items():
            if obj.lifetime == ContextLifetime.ONE_TURN:
                evicted.append(obj_id)
                continue
            replays = max(0, obj.expected_future_replays - 1)
            updated[obj_id] = ContextObject(
                id=obj.id,
                content=obj.content,
                tokens=obj.tokens,
                lifetime=obj.lifetime,
                created_turn=obj.created_turn,
                associated_hypothesis=obj.associated_hypothesis,
                associated_symbols=obj.associated_symbols,
                expected_future_replays=replays,
                decision_value=obj.decision_value,
            )
        self._objects = updated
        return evicted

    def on_event(self, event_type: str, payload: Any = None) -> list[str]:
        """Event-triggered eviction (e.g. hypothesis rejected, patch modified, test passed)."""
        evicted = []
        remaining = {}
        for obj_id, obj in self._objects.items():
            should_evict = False
            if event_type == "hypothesis_rejected" and obj.associated_hypothesis == str(payload):
                should_evict = True
            elif event_type == "test_passed" and obj.lifetime == ContextLifetime.UNTIL_TEST:
                should_evict = True
            elif event_type == "patch_changed" and payload:
                modified_syms = set(payload) if isinstance(payload, (list, tuple, set)) else {str(payload)}
                if any(sym in modified_syms for sym in obj.associated_symbols):
                    should_evict = True

            if should_evict:
                evicted.append(obj_id)
            else:
                remaining[obj_id] = obj

        self._objects = remaining
        return evicted


def evaluate_cache_aware_compaction_benefit(
    prefix_tokens: int,
    delta_tokens: int,
    remaining_turns: int,
    price_fresh: float = 3.0,
    price_cached: float = 0.3,
) -> float:
    """Evaluate: savings_future_replay - cost_cache_invalidation.

    savings_future_replay = delta_tokens * remaining_turns * price_cached
    cost_cache_invalidation = prefix_tokens * (price_fresh - price_cached)
    Returns net benefit in dollars (positive indicates compaction is cheaper).
    """
    savings_future_replay = float(delta_tokens * remaining_turns) * (price_cached / 1_000_000.0)
    cost_cache_invalidation = float(prefix_tokens) * ((price_fresh - price_cached) / 1_000_000.0)
    return savings_future_replay - cost_cache_invalidation


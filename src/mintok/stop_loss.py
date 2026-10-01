"""Stop-Loss Controller and Catastrophic Trajectory Scorer for MinTok.

Attacks pathological tail-latency and runaway token spend:
1. Stop-Loss Controller:
   - 3 unsuccessful patches without new evidence -> force inference reset / require new evidence.
   - Same observation expanded twice -> block expansion, require alternative evidence source.
   - Suite failed identically 3 times -> block patching, force diagnosis.
2. Catastrophic Trajectory Scorer:
   - waste_ratio = tokens_before_last_useful_action / oracle_tokens.
   - Pathological indicators: dead_turns, repeated_reads, repeated_failures,
     repeated_patches, unverified_patches, unchanged_observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class StopLossAction(str, Enum):
    """Intervention decision from the Stop-Loss controller."""

    ALLOW = "ALLOW"
    FORCE_EVIDENCE_GATHERING = "FORCE_EVIDENCE_GATHERING"
    BLOCK_EXPANSION_REDIRECT = "BLOCK_EXPANSION_REDIRECT"
    FORCE_DIAGNOSIS = "FORCE_DIAGNOSIS"
    FORCE_RESET = "FORCE_RESET"


@dataclass
class StopLossController:
    """Intervenes on pathological thashing and repetitive failure loops."""

    max_unsuccessful_patches_without_evidence: int = 3
    max_observation_expansions: int = 2
    max_identical_test_failures: int = 3

    consecutive_failed_patches: int = 0
    evidence_gathered_since_last_patch: bool = False
    observation_expansion_counts: dict[str, int] = field(default_factory=dict)
    identical_failure_signatures: dict[str, int] = field(default_factory=dict)
    last_failure_signature: str | None = None
    interventions_count: int = 0

    def record_evidence_gathered(self, action: str, details: str = "") -> None:
        """Mark that fresh diagnostic evidence (slice/caller/test traceback) was ingested."""
        if any(k in action.lower() for k in ("read", "slice", "caller", "query", "inspect", "diagnos")):
            self.evidence_gathered_since_last_patch = True

    def record_observation_expansion(self, observation_id: str) -> None:
        """Record an expansion of a virtualized observation handle."""
        cnt = self.observation_expansion_counts.get(observation_id, 0) + 1
        self.observation_expansion_counts[observation_id] = cnt

    def record_patch_attempt(self, success: bool, failure_signature: str | None = None) -> None:
        """Record an applied patch attempt outcome."""
        if success:
            self.consecutive_failed_patches = 0
            self.evidence_gathered_since_last_patch = False
            self.last_failure_signature = None
        else:
            self.consecutive_failed_patches += 1
            self.evidence_gathered_since_last_patch = False
            if failure_signature:
                sig_count = self.identical_failure_signatures.get(failure_signature, 0) + 1
                self.identical_failure_signatures[failure_signature] = sig_count
                self.last_failure_signature = failure_signature

    def evaluate_proposed_action(
        self,
        proposed_action: str,
        target: str | None = None,
    ) -> tuple[StopLossAction, str]:
        """Check if proposed action violates stop-loss safety invariants."""
        act_lower = proposed_action.lower()

        # Rule 1: 3 unsuccessful patches without new evidence -> force investigation
        if any(k in act_lower for k in ("patch", "apply_patch", "edit", "change")):
            if (
                self.consecutive_failed_patches >= self.max_unsuccessful_patches_without_evidence
                and not self.evidence_gathered_since_last_patch
            ):
                self.interventions_count += 1
                return (
                    StopLossAction.FORCE_EVIDENCE_GATHERING,
                    f"Stop-Loss: {self.consecutive_failed_patches} consecutive failed patches without new evidence. Investigation, slice, or test required before next edit.",
                )

        # Rule 2: Same observation expanded twice -> block expansion, redirect to alternate source
        if "expand" in act_lower and target:
            exp_count = self.observation_expansion_counts.get(target, 0)
            if exp_count >= self.max_observation_expansions:
                self.interventions_count += 1
                return (
                    StopLossAction.BLOCK_EXPANSION_REDIRECT,
                    f"Stop-Loss: Observation '{target}' expanded {exp_count} times already. Redirection to alternative evidence source required.",
                )

        # Rule 3: Suite failed identically 3 times -> don't patch again, force diagnosis
        if any(k in act_lower for k in ("patch", "apply_patch", "edit")):
            if self.last_failure_signature:
                sig_count = self.identical_failure_signatures.get(self.last_failure_signature, 0)
                if sig_count >= self.max_identical_test_failures:
                    self.interventions_count += 1
                    return (
                        StopLossAction.FORCE_DIAGNOSIS,
                        f"Stop-Loss: Suite failed with identical signature {sig_count} times. Root-cause diagnosis required before next patch attempt.",
                    )

        return StopLossAction.ALLOW, "Action permitted by stop-loss invariants."


@dataclass(frozen=True, slots=True)
class CatastrophicTrajectoryReport:
    """Metrics quantifying waste, dead turns, and runaway risks in a trajectory."""

    total_tokens: int
    oracle_tokens: int
    waste_ratio: float  # tokens_before_last_useful_action / oracle_tokens
    dead_turns: int
    repeated_reads: int
    repeated_failures: int
    repeated_patches: int
    unverified_patches: int
    unchanged_observations: int
    is_catastrophic: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "oracle_tokens": self.oracle_tokens,
            "waste_ratio": round(self.waste_ratio, 2),
            "dead_turns": self.dead_turns,
            "repeated_reads": self.repeated_reads,
            "repeated_failures": self.repeated_failures,
            "repeated_patches": self.repeated_patches,
            "unverified_patches": self.unverified_patches,
            "unchanged_observations": self.unchanged_observations,
            "is_catastrophic": self.is_catastrophic,
            "reasons": self.reasons,
        }


class CatastrophicTrajectoryScorer:
    """Analyzes trajectory events for pathological waste and tail runaway risks."""

    def __init__(self, waste_ratio_threshold: float = 4.0) -> None:
        self.waste_ratio_threshold = waste_ratio_threshold

    def evaluate_trajectory(
        self,
        turns: Sequence[dict[str, Any]],
        oracle_tokens: int = 5000,
    ) -> CatastrophicTrajectoryReport:
        """Evaluate a sequence of trajectory turns for waste and failure patterns."""
        total_tokens = sum(int(t.get("tokens", 0) or t.get("raw_tokens", 0)) for t in turns)
        useful_tokens_watermark = 0
        current_tokens_acc = 0

        read_targets: dict[str, int] = {}
        patch_hashes: dict[str, int] = {}
        failure_signatures: dict[str, int] = {}
        last_observation_hash = None

        dead_turns = 0
        repeated_reads = 0
        repeated_failures = 0
        repeated_patches = 0
        unverified_patches = 0
        unchanged_observations = 0

        pending_patch_unverified = False

        for t in turns:
            tok = int(t.get("tokens", 0) or t.get("raw_tokens", 0))
            current_tokens_acc += tok
            action = str(t.get("action", "")).lower()
            target = str(t.get("target", ""))
            output = str(t.get("output", "") or t.get("observation", ""))
            is_useful = bool(t.get("useful", False))

            if is_useful:
                useful_tokens_watermark = current_tokens_acc

            # Detect repeated reads
            if any(k in action for k in ("read", "cat", "view", "slice")):
                if target:
                    cnt = read_targets.get(target, 0)
                    if cnt > 0:
                        repeated_reads += 1
                        dead_turns += 1
                    read_targets[target] = cnt + 1

            # Detect patches and verification
            if any(k in action for k in ("patch", "edit")):
                if pending_patch_unverified:
                    unverified_patches += 1
                pending_patch_unverified = True
                if target:
                    pc = patch_hashes.get(target, 0)
                    if pc > 0:
                        repeated_patches += 1
                    patch_hashes[target] = pc + 1

            if any(k in action for k in ("verify", "test", "pytest")):
                pending_patch_unverified = False
                if "failed" in output.lower():
                    sig = output[:80]
                    fc = failure_signatures.get(sig, 0)
                    if fc > 0:
                        repeated_failures += 1
                    failure_signatures[sig] = fc + 1

            # Detect unchanged observations
            obs_hash = hash(output) if output else None
            if obs_hash is not None and obs_hash == last_observation_hash:
                unchanged_observations += 1
                dead_turns += 1
            last_observation_hash = obs_hash

        if pending_patch_unverified:
            unverified_patches += 1

        waste_tokens = useful_tokens_watermark if useful_tokens_watermark > 0 else total_tokens
        waste_ratio = float(waste_tokens) / max(1.0, float(oracle_tokens))

        reasons = []
        if waste_ratio > self.waste_ratio_threshold:
            reasons.append(f"Waste ratio ({waste_ratio:.2f}x) exceeds threshold ({self.waste_ratio_threshold}x)")
        if repeated_reads >= 3:
            reasons.append(f"Excessive repeated reads ({repeated_reads})")
        if repeated_failures >= 3:
            reasons.append(f"Repeated identical failures ({repeated_failures})")
        if unverified_patches >= 2:
            reasons.append(f"Unverified patch chain ({unverified_patches} unverified)")

        is_catastrophic = len(reasons) > 0 or (total_tokens > 100_000 and useful_tokens_watermark == 0)

        return CatastrophicTrajectoryReport(
            total_tokens=total_tokens,
            oracle_tokens=oracle_tokens,
            waste_ratio=waste_ratio,
            dead_turns=dead_turns,
            repeated_reads=repeated_reads,
            repeated_failures=repeated_failures,
            repeated_patches=repeated_patches,
            unverified_patches=unverified_patches,
            unchanged_observations=unchanged_observations,
            is_catastrophic=is_catastrophic,
            reasons=reasons,
        )

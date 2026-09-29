"""Failure Continuation Predictor & Clean-Context Restart for MinTok 3.1.

Attacks tail-latency and million-token runaway trajectories (e.g. openhands-resolver-137):
1. Continuation Predictor: Estimates P(solve within next 50k tokens), P(solve eventually).
   When P(solve) falls below threshold (e.g. 0.02 after 800k tokens), triggers early termination.
2. Clean-Context Restart: Rather than dragging a 25-turn contaminated conversation history,
   distills current state into a minimal clean-slate restart packet:
   - Preserves: task goal, verified facts, repo profile, current best patch, test targets.
   - Discards: intermediate conversational prose, circular reasoning, stale hypotheses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mintok.conversation import CanonicalState, EvidenceFact, FactResidencyTier
from mintok.repo_profile import RepoProfile


@dataclass(frozen=True, slots=True)
class ContinuationEstimate:
    """Predicted likelihood of task resolution over upcoming token horizons."""

    tokens_spent: int
    turns_elapsed: int
    p_solve_next_50k: float
    p_solve_next_100k: float
    p_solve_eventually: float
    recommended_action: str  # "CONTINUE", "RESTART_FROM_CHECKPOINT", "ABORT_TASK"
    reason: str


class ContinuationPredictor:
    """Predicts continuation utility to terminate unproductive trajectories early."""

    def __init__(
        self,
        token_ceiling: int = 1_500_000,
        min_continuation_prob: float = 0.03,
    ) -> None:
        self.token_ceiling = token_ceiling
        self.min_continuation_prob = min_continuation_prob

    def evaluate(
        self,
        tokens_spent: int,
        turns_elapsed: int,
        consecutive_failures: int = 0,
        tests_resolved: int = 0,
        has_patch: bool = False,
    ) -> ContinuationEstimate:
        """Estimate remaining solve probability and determine optimal trajectory action."""
        # Baseline solve probability decays as turns and spend accumulate without progress
        base_decay = math.exp(-float(tokens_spent) / 450_000.0)
        failure_penalty = math.exp(-0.35 * max(0, consecutive_failures))
        progress_boost = 1.0 + (0.4 * min(3, tests_resolved)) if tests_resolved > 0 else 0.8
        patch_boost = 1.2 if has_patch else 0.9

        p_eventual = min(0.95, max(0.005, 0.45 * base_decay * failure_penalty * progress_boost * patch_boost))
        p_next_50k = min(p_eventual, p_eventual * 0.45)
        p_next_100k = min(p_eventual, p_eventual * 0.75)

        # Decision policy
        if tokens_spent >= self.token_ceiling or (tokens_spent > 600_000 and p_next_100k < self.min_continuation_prob):
            action = "ABORT_TASK"
            reason = f"tokens ({tokens_spent:,}) exceed economical threshold; P(solve)={p_next_100k:.3f}"
        elif turns_elapsed >= 10 and consecutive_failures >= 4 and not tests_resolved:
            action = "RESTART_FROM_CHECKPOINT"
            reason = f"agent thrashing across {turns_elapsed} turns ({consecutive_failures} consecutive failures)"
        else:
            action = "CONTINUE"
            reason = f"healthy progression; P(solve_next_100k)={p_next_100k:.3f}"

        return ContinuationEstimate(
            tokens_spent=tokens_spent,
            turns_elapsed=turns_elapsed,
            p_solve_next_50k=round(p_next_50k, 4),
            p_solve_next_100k=round(p_next_100k, 4),
            p_solve_eventually=round(p_eventual, 4),
            recommended_action=action,
            reason=reason,
        )


class CleanContextRestart:
    """Extracts verified invariants into a clean-slate session prompt."""

    @staticmethod
    def build_restart_prompt(
        state: CanonicalState,
        task_instruction: str,
        repo_profile: RepoProfile | None = None,
    ) -> str:
        """Create a zero-baggage prompt containing only verified ground truth."""
        lines = [
            "### FRESH CLEAN CONTEXT RESTART",
            "Previous exploratory reasoning has been purged. Continue with verified facts below:",
            f"**Task Goal:** {task_instruction}",
        ]

        if repo_profile:
            lines.append(f"**Repo:** {repo_profile.repo_name} (tests: {repo_profile.primary_test_runner})")

        # Keep strictly HOT/WARM verified facts
        hot_facts = [
            f.render() if isinstance(f, EvidenceFact) else str(f)
            for f in state.verified_facts
            if not isinstance(f, EvidenceFact) or f.residency != FactResidencyTier.COLD
        ]
        if hot_facts:
            lines.append("**Verified Ground Truth:**")
            for hf in hot_facts:
                lines.append(f"- {hf}")

        if state.current_patch:
            lines.append(f"**Current Validated Patch:**\n```\n{state.current_patch}\n```")

        if state.current_failures:
            lines.append(f"**Target Failing Tests:** {', '.join(state.current_failures)}")

        lines.append("Next Step: Analyze verified facts above and emit targeted fix without prior loop bias.")
        return "\n".join(lines)

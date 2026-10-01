"""Learned Local Policy Predictor and Context Allocator for MinTok.

Scores candidate agent actions at decision states using:
    Score = P(solve | action, state) * V - lambda * E[token_cost | action, state] - E[recovery_cost | action, state]

Features:
- Pure stdlib implementation (zero third-party dependencies).
- Feature extraction over AgentState (turn, tokens spent, failures, patch lines, repo complexity, task family).
- Action modeling across core action types:
  read_full, read_slice, query_symbol, virtualize_output, compact_state, run_verifier, apply_patch, inspect_bundle.
- Offline parameter calibration from historical trajectory records.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class AgentState:
    """Snapshot of agent conversation and environment state."""

    turn: int = 1
    tokens_spent: int = 0
    active_failures: int = 1
    patch_lines: int = 0
    repo_complexity: float = 0.5  # 0.0 to 1.0
    task_family: str = "general"
    has_uninspected_output: bool = False
    context_window_usage: float = 0.1  # 0.0 to 1.0

    # Extended GBDT Tabular Feature set
    task_loc: int = 150
    target_loc: int = 45
    num_files: int = 8
    ast_symbols_count: int = 24
    fan_in: int = 3
    fan_out: int = 4
    call_graph_depth: int = 3
    candidate_dispersion: float = 0.2
    likely_callers_count: int = 2
    test_availability: bool = True
    failure_signature: str = ""
    previous_tool_count: int = 3
    current_context_tokens: int = 24000
    recent_expansion_count: int = 0
    recent_patch_count: int = 1
    suite_status: str = "passing"

    def to_feature_vector(self) -> list[float]:
        """Convert state features into a normalized numerical vector for tree-based tabular models."""
        return [
            float(self.task_loc) / 500.0,
            float(self.target_loc) / 200.0,
            float(self.num_files),
            float(self.ast_symbols_count),
            float(self.fan_in),
            float(self.fan_out),
            float(self.call_graph_depth),
            float(self.candidate_dispersion),
            float(self.likely_callers_count),
            1.0 if self.test_availability else 0.0,
            float(self.previous_tool_count),
            float(self.current_context_tokens) / 100_000.0,
            float(self.recent_expansion_count),
            float(self.recent_patch_count),
            float(self.repo_complexity),
            float(self.turn),
            float(self.active_failures),
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateAction:
    """A proposed action at a decision point."""

    action_type: str  # read_full, read_slice, query_symbol, virtualize_output, compact_state, run_verifier, apply_patch, inspect_bundle
    target: str = ""
    raw_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionScore:
    """Predicted outcome, cost, and utility score for a candidate action."""

    action: CandidateAction
    p_solve: float
    expected_tokens: float
    expected_recovery: float
    utility_score: float
    expected_turns: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "p_solve": round(self.p_solve, 4),
            "expected_tokens": round(self.expected_tokens, 1),
            "expected_turns": round(self.expected_turns, 2),
            "expected_recovery": round(self.expected_recovery, 1),
            "utility_score": round(self.utility_score, 4),
        }


class PolicyPredictor:
    """Predictor and allocator estimating P(solve), E[tokens], E[recovery] and expected utility."""

    def __init__(
        self,
        base_p_solve: float = 0.70,
        solve_value: float = 1.0,
        token_lambda: float = 0.000005,
    ) -> None:
        self.base_p_solve = base_p_solve
        self.solve_value = solve_value
        self.token_lambda = token_lambda

        # Action-specific intrinsic properties
        self.action_token_factors: dict[str, float] = {
            "read_full": 1.00,
            "read_slice": 0.22,
            "query_symbol": 0.08,
            "virtualize_output": 0.05,
            "compact_state": 0.15,
            "run_verifier": 0.35,
            "apply_patch": 0.18,
            "inspect_bundle": 0.12,
        }

        self.action_solve_deltas: dict[str, float] = {
            "read_full": -0.02,  # context pollution risk on large repos
            "read_slice": 0.05,
            "query_symbol": 0.04,
            "virtualize_output": 0.02,
            "compact_state": 0.03,
            "run_verifier": 0.08,
            "apply_patch": 0.04,
            "inspect_bundle": 0.06,
        }

        self.action_recovery_factors: dict[str, float] = {
            "read_full": 0.0,
            "read_slice": 0.05,
            "query_symbol": 0.02,
            "virtualize_output": 0.03,
            "compact_state": 0.04,
            "run_verifier": 0.0,
            "apply_patch": 0.0,
            "inspect_bundle": 0.01,
        }

    def predict_p_solve(self, state: AgentState, action: CandidateAction) -> float:
        """Estimate P(solve | action, state)."""
        base = self.base_p_solve + self.action_solve_deltas.get(action.action_type, 0.0)

        # Context bloat penalty: high token usage degrades solve rate
        if state.tokens_spent > 60000 or state.context_window_usage > 0.75:
            if action.action_type == "read_full":
                base -= 0.15
            elif action.action_type in ("compact_state", "virtualize_output"):
                base += 0.08

        # Verification benefit when failures exist
        if state.active_failures > 0 and action.action_type == "run_verifier":
            base += 0.06

        # Complex repo adjustments
        if state.repo_complexity > 0.7:
            if action.action_type == "read_full":
                base -= 0.10
            elif action.action_type in ("inspect_bundle", "query_symbol"):
                base += 0.07

        # Clamp between 0.01 and 0.99
        return max(0.01, min(0.99, base))

    def predict_expected_tokens(self, state: AgentState, action: CandidateAction) -> float:
        """Estimate E[tokens | action, state]."""
        raw = action.raw_tokens if action.raw_tokens > 0 else 2500
        factor = self.action_token_factors.get(action.action_type, 1.0)
        est = raw * factor

        # Scaling with repo complexity
        if action.action_type == "read_full":
            est *= (1.0 + state.repo_complexity * 2.0)
        return float(est)

    def predict_expected_recovery_cost(self, state: AgentState, action: CandidateAction) -> float:
        """Estimate E[recovery | action, state] in utility penalty units."""
        base_factor = self.action_recovery_factors.get(action.action_type, 0.0)
        # If output was heavily virtualized and repo is complex, slight recovery risk
        penalty = base_factor * (1.0 + state.repo_complexity) * 0.05
        return float(penalty)

    def predict_expected_turns(self, state: AgentState, action: CandidateAction) -> float:
        """Estimate additional turns required given action."""
        if action.action_type in ("apply_patch", "run_verifier"):
            return 1.2
        if action.action_type in ("read_slice", "query_symbol"):
            return 2.5
        if action.action_type == "read_full":
            return 3.5
        return 2.0

    def score_action(
        self,
        state: AgentState,
        action: CandidateAction,
        value: float | None = None,
        token_lambda: float | None = None,
    ) -> ActionScore:
        """Compute net utility: Score = P(solve)*V - lambda*E[tokens] - E[recovery]."""
        v = self.solve_value if value is None else value
        lmb = self.token_lambda if token_lambda is None else token_lambda

        p_solve = self.predict_p_solve(state, action)
        exp_tokens = self.predict_expected_tokens(state, action)
        exp_turns = self.predict_expected_turns(state, action)
        exp_recovery = self.predict_expected_recovery_cost(state, action)

        score = (p_solve * v) - (lmb * exp_tokens) - exp_recovery

        return ActionScore(
            action=action,
            p_solve=p_solve,
            expected_tokens=exp_tokens,
            expected_turns=exp_turns,
            expected_recovery=exp_recovery,
            utility_score=score,
        )

    def rank_actions(
        self,
        state: AgentState,
        actions: Sequence[CandidateAction],
        value: float | None = None,
        token_lambda: float | None = None,
    ) -> list[ActionScore]:
        """Rank candidate actions by utility score descending."""
        scored = [self.score_action(state, a, value, token_lambda) for a in actions]
        return sorted(scored, key=lambda s: s.utility_score, reverse=True)

    def select_best_action(
        self,
        state: AgentState,
        actions: Sequence[CandidateAction],
        value: float | None = None,
        token_lambda: float | None = None,
    ) -> ActionScore:
        """Select the candidate action with highest utility score."""
        ranked = self.rank_actions(state, actions, value, token_lambda)
        if not ranked:
            raise ValueError("Cannot select from empty actions list")
        return ranked[0]

    def fit(self, records: Sequence[dict[str, Any]]) -> None:
        """Calibrate action token multipliers and solve deltas against historical records."""
        if not records:
            return

        action_counts: dict[str, int] = {}
        action_solves: dict[str, int] = {}
        action_tokens: dict[str, int] = {}

        for r in records:
            act = r.get("action_type") or r.get("action", "")
            if not act:
                continue
            action_counts[act] = action_counts.get(act, 0) + 1
            if r.get("solved", False):
                action_solves[act] = action_solves.get(act, 0) + 1
            action_tokens[act] = action_tokens.get(act, 0) + int(r.get("tokens", 0))

        # Update empirical solve deltas if sufficient sample
        for act, cnt in action_counts.items():
            if cnt >= 5:
                emp_p = action_solves.get(act, 0) / cnt
                self.action_solve_deltas[act] = round(emp_p - self.base_p_solve, 4)

    def export_weights(self) -> dict[str, Any]:
        """Export current model parameters."""
        return {
            "base_p_solve": self.base_p_solve,
            "solve_value": self.solve_value,
            "token_lambda": self.token_lambda,
            "action_token_factors": dict(self.action_token_factors),
            "action_solve_deltas": dict(self.action_solve_deltas),
            "action_recovery_factors": dict(self.action_recovery_factors),
        }

    def load_weights(self, weights: dict[str, Any]) -> None:
        """Load model parameters."""
        self.base_p_solve = weights.get("base_p_solve", self.base_p_solve)
        self.solve_value = weights.get("solve_value", self.solve_value)
        self.token_lambda = weights.get("token_lambda", self.token_lambda)
        if "action_token_factors" in weights:
            self.action_token_factors.update(weights["action_token_factors"])
        if "action_solve_deltas" in weights:
            self.action_solve_deltas.update(weights["action_solve_deltas"])
        if "action_recovery_factors" in weights:
            self.action_recovery_factors.update(weights["action_recovery_factors"])

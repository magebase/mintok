"""Action-Value Model (Value of an Observation), Uncertainty Decomposition, Survival Analysis, and Trajectory Fingerprints.

Enables MinTok to operate as an inference optimization coprocessor:
1. Value of an Observation:
   - Models V(s, a) = E[future tokens saved + Delta P(success)] rather than myopic single-step prediction.
2. Rejected Alternatives Decision Logging:
   - Logs chosen action vs all rejected candidate actions with utilities to train pairwise rankings rank(a1, a2 | s).
3. Uncertainty Decomposition:
   - Decomposes agent uncertainty into: task_uncertainty, repo_uncertainty, hypothesis_uncertainty,
     patch_uncertainty, and verification_uncertainty.
4. Early Trajectory Failure Prediction (Survival Analysis):
   - Estimates P(solve | trajectory_1:t) and hazard P(runaway > T) to trigger early strategy pivots before wasting tokens.
5. Trajectory Archetype Fingerprints:
   - Classifies trajectory behavior into archetypes: EFFICIENT_SOLVER, OVER_READER, PATCH_THRASHING,
     PREMATURE_PATCHER, VERIFIER_HEAVY, HYPOTHESIS_WANDERER.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from mintok.hypothesis import HypothesisGraph


class TrajectoryArchetype(str, Enum):
    """Behavioral fingerprint of an agent trajectory."""

    EFFICIENT_SOLVER = "EFFICIENT_SOLVER"
    OVER_READER = "OVER_READER"
    PATCH_THRASHING = "PATCH_THRASHING"
    PREMATURE_PATCHER = "PREMATURE_PATCHER"
    VERIFIER_HEAVY = "VERIFIER_HEAVY"
    HYPOTHESIS_WANDERER = "HYPOTHESIS_WANDERER"


@dataclass(frozen=True, slots=True)
class ObservationValueEstimate:
    """Downstream value assessment for a proposed observation or action."""

    action_type: str
    target: str
    direct_token_cost: int
    hypotheses_entropy_reduction: float
    estimated_future_tokens_saved: int
    downstream_p_solve_gain: float
    net_observation_value: float  # Future tokens saved + value * delta_P - direct_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "direct_token_cost": self.direct_token_cost,
            "hypotheses_entropy_reduction": round(self.hypotheses_entropy_reduction, 4),
            "estimated_future_tokens_saved": self.estimated_future_tokens_saved,
            "downstream_p_solve_gain": round(self.downstream_p_solve_gain, 4),
            "net_observation_value": round(self.net_observation_value, 4),
        }


class ActionValueModel:
    """Estimates the value of an observation V(s, a) = E[future tokens saved + Delta P(success)]."""

    def __init__(
        self,
        token_cost_weight: float = 0.0001,
        success_weight: float = 1.0,
    ) -> None:
        self.token_cost_weight = token_cost_weight
        self.success_weight = success_weight

    def evaluate_observation_value(
        self,
        action_type: str,
        target: str,
        hypothesis_graph: HypothesisGraph | None = None,
        base_tokens: int = 1500,
    ) -> ObservationValueEstimate:
        """Estimate downstream value gained by acquiring this observation."""
        act_lower = action_type.lower()

        # 1. Direct cost estimation
        if "slice" in act_lower:
            direct_cost = int(base_tokens * 0.25)
        elif "symbol" in act_lower or "caller" in act_lower:
            direct_cost = int(base_tokens * 0.15)
        elif "test" in act_lower or "verifier" in act_lower:
            direct_cost = int(base_tokens * 0.40)
        elif "full" in act_lower or "read" in act_lower:
            direct_cost = base_tokens
        else:
            direct_cost = int(base_tokens * 0.20)

        # 2. Discriminating power and entropy reduction
        entropy_reduction = 0.0
        if hypothesis_graph is not None:
            prior_entropy = hypothesis_graph.compute_entropy()
            discrim = hypothesis_graph.evaluate_discriminating_power(action_type)
            entropy_reduction = prior_entropy * discrim * 0.65
        else:
            if "caller" in act_lower or "symbol" in act_lower:
                entropy_reduction = 0.55
            elif "slice" in act_lower:
                entropy_reduction = 0.45
            else:
                entropy_reduction = 0.20

        # 3. Future tokens saved (pruning false search branches saves 3,000 to 12,000 tokens)
        future_tokens_saved = int(entropy_reduction * 8000)

        # 4. Downstream success probability gain
        p_solve_gain = min(0.35, entropy_reduction * 0.40)

        # 5. Net observation value
        net_val = (
            (float(future_tokens_saved) * self.token_cost_weight)
            + (p_solve_gain * self.success_weight)
            - (float(direct_cost) * self.token_cost_weight)
        )

        return ObservationValueEstimate(
            action_type=action_type,
            target=target,
            direct_token_cost=direct_cost,
            hypotheses_entropy_reduction=entropy_reduction,
            estimated_future_tokens_saved=future_tokens_saved,
            downstream_p_solve_gain=p_solve_gain,
            net_observation_value=net_val,
        )


@dataclass(frozen=True, slots=True)
class RejectedAlternative:
    """Candidate action evaluated but not chosen."""

    action_type: str
    target: str
    utility_score: float
    p_solve: float
    expected_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "utility_score": round(self.utility_score, 4),
            "p_solve": round(self.p_solve, 4),
            "expected_tokens": self.expected_tokens,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Decision snapshot recording chosen action and all rejected alternatives for ranking."""

    turn: int
    chosen_action: str
    chosen_target: str
    chosen_utility: float
    rejected_alternatives: list[RejectedAlternative]
    predicted_p_solve: float
    predicted_tokens: int
    actual_tokens: int = 0
    eventual_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "chosen_action": self.chosen_action,
            "chosen_target": self.chosen_target,
            "chosen_utility": round(self.chosen_utility, 4),
            "rejected_alternatives": [a.to_dict() for a in self.rejected_alternatives],
            "predicted_p_solve": round(self.predicted_p_solve, 4),
            "predicted_tokens": self.predicted_tokens,
            "actual_tokens": self.actual_tokens,
            "eventual_success": self.eventual_success,
        }


class AlternativesLogger:
    """Logs decisions and rejected candidate alternatives for ranking optimization."""

    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []

    def log_decision(
        self,
        turn: int,
        chosen_action: str,
        chosen_target: str,
        chosen_utility: float,
        candidates: Sequence[dict[str, Any]],
        predicted_p_solve: float,
        predicted_tokens: int,
        actual_tokens: int = 0,
        eventual_success: bool | None = None,
    ) -> DecisionRecord:
        """Record the selected action alongside rejected options."""
        rejected = []
        for c in candidates:
            act = c.get("action_type") or c.get("action", "")
            tgt = c.get("target", "")
            if act == chosen_action and tgt == chosen_target:
                continue
            rejected.append(
                RejectedAlternative(
                    action_type=act,
                    target=tgt,
                    utility_score=float(c.get("utility_score", c.get("utility", 0.0))),
                    p_solve=float(c.get("p_solve", 0.5)),
                    expected_tokens=int(c.get("expected_tokens", 1000)),
                )
            )

        rec = DecisionRecord(
            turn=turn,
            chosen_action=chosen_action,
            chosen_target=chosen_target,
            chosen_utility=chosen_utility,
            rejected_alternatives=rejected,
            predicted_p_solve=predicted_p_solve,
            predicted_tokens=predicted_tokens,
            actual_tokens=actual_tokens,
            eventual_success=eventual_success,
        )
        self.records.append(rec)
        return rec

    def get_records(self) -> list[DecisionRecord]:
        return list(self.records)


@dataclass(frozen=True, slots=True)
class UncertaintyBreakdown:
    """Disaggregated uncertainty components."""

    task_uncertainty: float  # ambiguity in problem statement (0.0 to 1.0)
    repo_uncertainty: float  # codebase complexity & dispersion (0.0 to 1.0)
    hypothesis_uncertainty: float  # entropy over root causes (0.0 to 1.0)
    patch_uncertainty: float  # uncertainty over required modification (0.0 to 1.0)
    verification_uncertainty: float  # uncertainty over test coverage (0.0 to 1.0)
    total_uncertainty: float
    dominant_factor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_uncertainty": round(self.task_uncertainty, 4),
            "repo_uncertainty": round(self.repo_uncertainty, 4),
            "hypothesis_uncertainty": round(self.hypothesis_uncertainty, 4),
            "patch_uncertainty": round(self.patch_uncertainty, 4),
            "verification_uncertainty": round(self.verification_uncertainty, 4),
            "total_uncertainty": round(self.total_uncertainty, 4),
            "dominant_factor": self.dominant_factor,
        }


class UncertaintyDecomposer:
    """Decomposes overall agent uncertainty into actionable sub-dimensions."""

    @staticmethod
    def decompose(
        task_ambiguity: float = 0.3,
        repo_complexity: float = 0.5,
        hypothesis_entropy: float = 0.6,
        patch_lines: int = 0,
        tests_available: bool = True,
    ) -> UncertaintyBreakdown:
        task_u = max(0.0, min(1.0, task_ambiguity))
        repo_u = max(0.0, min(1.0, repo_complexity))
        hyp_u = max(0.0, min(1.0, hypothesis_entropy / 2.0))
        patch_u = 0.7 if patch_lines == 0 else max(0.1, min(0.9, float(patch_lines) / 100.0))
        verif_u = 0.2 if tests_available else 0.85

        factors = {
            "task_uncertainty": task_u,
            "repo_uncertainty": repo_u,
            "hypothesis_uncertainty": hyp_u,
            "patch_uncertainty": patch_u,
            "verification_uncertainty": verif_u,
        }

        total = sum(factors.values()) / float(len(factors))
        dominant = max(factors.keys(), key=lambda k: factors[k])

        return UncertaintyBreakdown(
            task_uncertainty=task_u,
            repo_uncertainty=repo_u,
            hypothesis_uncertainty=hyp_u,
            patch_uncertainty=patch_u,
            verification_uncertainty=verif_u,
            total_uncertainty=total,
            dominant_factor=dominant,
        )


class SurvivalPredictor:
    """Estimates trajectory survival, solve probability, and runaway hazard."""

    @staticmethod
    def evaluate(
        turn: int,
        tokens_spent: int,
        failed_patches: int,
        repeated_actions: int = 0,
        tests_passing: bool = False,
    ) -> tuple[float, float, str]:
        """Estimate (p_solve, runaway_hazard, recommendation).
        
        Recommendation: 'CONTINUE' | 'PIVOT_STRATEGY' | 'EARLY_ABORT'
        """
        if tests_passing:
            return 0.98, 0.01, "CONTINUE"

        # Baseline solve decay
        base_p = 0.75
        # Token burn penalty
        token_burn = float(tokens_spent) / 100_000.0
        # Patch failures penalty
        patch_penalty = float(failed_patches) * 0.18
        # Repetition penalty
        rep_penalty = float(repeated_actions) * 0.10
        # Turn penalty
        turn_penalty = float(turn) * 0.02

        p_solve = max(0.02, min(0.95, base_p - token_burn - patch_penalty - rep_penalty - turn_penalty))

        # Hazard modeling: probability of endless runaway or catastrophic waste
        hazard = min(0.99, (patch_penalty * 1.5) + (rep_penalty * 1.2) + (token_burn * 0.8))

        if hazard >= 0.75 or (failed_patches >= 3 and tokens_spent > 40000):
            recommendation = "EARLY_ABORT"
        elif hazard >= 0.45 or failed_patches >= 2 or repeated_actions >= 2:
            recommendation = "PIVOT_STRATEGY"
        else:
            recommendation = "CONTINUE"

        return round(p_solve, 4), round(hazard, 4), recommendation


class TrajectoryFingerprinter:
    """Classifies trajectory into archetypes and suggests controller interventions."""

    @staticmethod
    def classify(
        action_sequence: Sequence[str],
        tokens_spent: int,
        failed_patches: int,
        verification_count: int,
    ) -> tuple[TrajectoryArchetype, float, str]:
        """Classify behavior archetype and return (archetype, confidence, intervention)."""
        n = len(action_sequence)
        if n == 0:
            return TrajectoryArchetype.EFFICIENT_SOLVER, 0.5, "Standard progression"

        reads = sum(1 for a in action_sequence if any(k in a.lower() for k in ("read", "slice", "cat", "view")))
        patches = sum(1 for a in action_sequence if any(k in a.lower() for k in ("patch", "edit")))
        verifs = sum(1 for a in action_sequence if any(k in a.lower() for k in ("test", "verify", "pytest")))

        read_ratio = float(reads) / float(n)
        patch_ratio = float(patches) / float(n)

        # 1. Patch thrashing: multiple patches with high failure rate
        if failed_patches >= 2 and patch_ratio > 0.35:
            return (
                TrajectoryArchetype.PATCH_THRASHING,
                0.90,
                "Enforce stop-loss: lock code edits and mandate root-cause traceback diagnosis.",
            )

        # 2. Over-reader: high proportion of reads without patches
        if reads >= 3 and patches == 0 and tokens_spent > 15000:
            return (
                TrajectoryArchetype.OVER_READER,
                0.85,
                "Cap read sizes; enforce pre-action token budgets and require evidence synthesis.",
            )

        # 3. Premature patcher: patch applied on turn 1 or 2 without gathering caller or context
        if patches >= 1 and reads <= 1 and n <= 2:
            return (
                TrajectoryArchetype.PREMATURE_PATCHER,
                0.80,
                "Intercept early patch; verify caller constraints and blast radius before applying change.",
            )

        # 4. Verifier heavy: repeated test invocations with minimal editing
        if verifs >= 3 and patches <= 1:
            return (
                TrajectoryArchetype.VERIFIER_HEAVY,
                0.80,
                "Throttle test execution; switch to focused unit verification.",
            )

        # 5. Hypothesis wanderer: high turns without settling on a target
        if n >= 4 and reads >= 2 and patches <= 1 and failed_patches == 0 and tokens_spent > 25000:
            return (
                TrajectoryArchetype.HYPOTHESIS_WANDERER,
                0.75,
                "Prune hypothesis tree; focus inquiry on highest-confidence candidate discriminator.",
            )

        return (
            TrajectoryArchetype.EFFICIENT_SOLVER,
            0.85,
            "Trajectory behaving efficiently within expected token and turn parameters.",
        )

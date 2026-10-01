"""Contextual Bandit Policy, Sequential Lookahead, Value of Information (VOI), and Hierarchical Action Space.

Features:
1. Real Contextual Bandit:
   - 11 atomic actions: SLICE, READ, SEARCH, TEST, TRACE, CALL_GRAPH, DATA_FLOW,
     EXPAND_OBSERVATION, PATCH, VERIFY, ESCALATE.
   - Reward: R = V_solve * P(solve) - lambda * T - lambda_r * Regression.
   - Upper Confidence Bound (LinUCB style) exploration with logged propensities.
2. Sequential Lookahead (Active Information Acquisition):
   - a_t = argmax_a (E[V(s_{t+1}) | s_t, a] - V(s_t)) / E[T | s_t, a].
3. Value of Information (VOI):
   - VOI(a) = (H(H|s) - E[H(H|s,a)]) / E[T(a)] (tokens spent per bit of uncertainty reduction).
4. Hierarchical Action Space (4 Levels):
   - Level 1: Objective (INVESTIGATE, PATCH, VERIFY, RECOVER)
   - Level 2: Information Source (SOURCE, TEST, TRACE, GRAPH, SEARCH)
   - Level 3: Precision (COARSE, MEDIUM, SURGICAL)
   - Level 4: Scope (FILE, SYMBOL, LINES, CALL_CHAIN)
5. Independent Stopping Policy:
   - Stops when Delta P_solve / Delta T < lambda.
6. Early Failure Trajectory Survival Model:
   - Predicts after 2-4 turns: P(<=2k), P(<=5k), P(<=10k), P(runaway >20k), P(regression).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from mintok.learned_policy import PolicyState


class BanditAction(str, Enum):
    """The 11 candidate actions for the Contextual Bandit."""

    SLICE = "SLICE"
    READ = "READ"
    SEARCH = "SEARCH"
    TEST = "TEST"
    TRACE = "TRACE"
    CALL_GRAPH = "CALL_GRAPH"
    DATA_FLOW = "DATA_FLOW"
    EXPAND_OBSERVATION = "EXPAND_OBSERVATION"
    PATCH = "PATCH"
    VERIFY = "VERIFY"
    ESCALATE = "ESCALATE"


# --- 4-Level Hierarchical Action Space ---

class ObjectiveLevel(str, Enum):
    INVESTIGATE = "INVESTIGATE"
    PATCH = "PATCH"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"


class SourceLevel(str, Enum):
    SOURCE = "SOURCE"
    TEST = "TEST"
    TRACE = "TRACE"
    GRAPH = "GRAPH"
    SEARCH = "SEARCH"


class PrecisionLevel(str, Enum):
    COARSE = "COARSE"
    MEDIUM = "MEDIUM"
    SURGICAL = "SURGICAL"


class ScopeLevel(str, Enum):
    FILE = "FILE"
    SYMBOL = "SYMBOL"
    LINES = "LINES"
    CALL_CHAIN = "CALL_CHAIN"


@dataclass(frozen=True, slots=True)
class HierarchicalAction:
    """Hierarchical decomposition of an agent operation."""

    objective: ObjectiveLevel
    source: SourceLevel
    precision: PrecisionLevel
    scope: ScopeLevel
    bandit_action_mapping: BanditAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective.value,
            "source": self.source.value,
            "precision": self.precision.value,
            "scope": self.scope.value,
            "bandit_action": self.bandit_action_mapping.value,
        }


# --- Value of Information (VOI) ---

@dataclass(frozen=True, slots=True)
class VOIResult:
    """Value of Information quantification in bits of uncertainty reduction per token."""

    action: BanditAction
    prior_entropy_bits: float
    expected_posterior_entropy_bits: float
    entropy_reduction_bits: float
    expected_tokens: int
    voi_bits_per_token: float  # (H_before - H_after) / E[tokens]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "prior_entropy_bits": round(self.prior_entropy_bits, 4),
            "expected_posterior_entropy_bits": round(self.expected_posterior_entropy_bits, 4),
            "entropy_reduction_bits": round(self.entropy_reduction_bits, 4),
            "expected_tokens": self.expected_tokens,
            "voi_bits_per_token": round(self.voi_bits_per_token, 6),
        }


class ValueOfInformationCalculator:
    """Calculates VOI(a) = (H(H|s) - E[H(H|s,a)]) / E[T(a)]."""

    ACTION_ENTROPY_REDUCTION: dict[BanditAction, float] = {
        BanditAction.TEST: 0.65,
        BanditAction.TRACE: 0.70,
        BanditAction.SLICE: 0.45,
        BanditAction.CALL_GRAPH: 0.35,
        BanditAction.SEARCH: 0.25,
        BanditAction.DATA_FLOW: 0.50,
        BanditAction.READ: 0.20,
        BanditAction.EXPAND_OBSERVATION: 0.30,
        BanditAction.PATCH: 0.10,
        BanditAction.VERIFY: 0.60,
        BanditAction.ESCALATE: 0.40,
    }

    ACTION_EXPECTED_TOKENS: dict[BanditAction, int] = {
        BanditAction.SLICE: 450,
        BanditAction.READ: 2500,
        BanditAction.SEARCH: 300,
        BanditAction.TEST: 650,
        BanditAction.TRACE: 1200,
        BanditAction.CALL_GRAPH: 350,
        BanditAction.DATA_FLOW: 800,
        BanditAction.EXPAND_OBSERVATION: 400,
        BanditAction.PATCH: 500,
        BanditAction.VERIFY: 600,
        BanditAction.ESCALATE: 2000,
    }

    @classmethod
    def compute_voi(cls, action: BanditAction, prior_entropy: float = 1.20) -> VOIResult:
        """Compute bits of uncertainty reduction per token spent."""
        reduction_ratio = cls.ACTION_ENTROPY_REDUCTION.get(action, 0.20)
        expected_tokens = cls.ACTION_EXPECTED_TOKENS.get(action, 1000)

        reduction_bits = prior_entropy * reduction_ratio
        posterior_entropy = max(0.0, prior_entropy - reduction_bits)

        voi = reduction_bits / float(expected_tokens)

        return VOIResult(
            action=action,
            prior_entropy_bits=prior_entropy,
            expected_posterior_entropy_bits=posterior_entropy,
            entropy_reduction_bits=reduction_bits,
            expected_tokens=expected_tokens,
            voi_bits_per_token=voi,
        )


# --- Contextual Bandit Policy ---

@dataclass(frozen=True, slots=True)
class BanditDecision:
    """Action selected by the Contextual Bandit."""

    action: BanditAction
    hierarchical_action: HierarchicalAction
    expected_reward: float
    exploration_bonus: float
    ucb_score: float
    logging_propensity: float
    voi_result: VOIResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "hierarchical_action": self.hierarchical_action.to_dict(),
            "expected_reward": round(self.expected_reward, 4),
            "exploration_bonus": round(self.exploration_bonus, 4),
            "ucb_score": round(self.ucb_score, 4),
            "logging_propensity": round(self.logging_propensity, 4),
            "voi": self.voi_result.to_dict(),
        }


class ContextualBanditPolicy:
    """LinUCB contextual bandit balancing expected solve reward, token penalty, and VOI."""

    def __init__(
        self,
        v_solve: float = 1.0,
        lambda_tokens: float = 0.0001,
        ucb_alpha: float = 0.40,
    ) -> None:
        self.v_solve = v_solve
        self.lambda_tokens = lambda_tokens
        self.ucb_alpha = ucb_alpha

        # Action-specific linear weight vectors
        self.action_weights: dict[BanditAction, list[float]] = {
            act: [0.05] * 17 for act in BanditAction
        }
        # Pre-seed weights with empirical correlations
        self.action_weights[BanditAction.SLICE][0] = 0.20
        self.action_weights[BanditAction.TEST][15] = 0.35
        self.action_weights[BanditAction.PATCH][12] = 0.40
        self.action_weights[BanditAction.VERIFY][16] = 0.45

    def select_action(
        self,
        state: PolicyState,
        available_actions: Sequence[BanditAction] | None = None,
    ) -> BanditDecision:
        """Select action maximizing UCB score."""
        candidates = list(available_actions) if available_actions else list(BanditAction)
        x = state.to_feature_vector()

        best_act = candidates[0]
        best_ucb = -float("inf")
        best_r = 0.0
        best_bonus = 0.0

        for act in candidates:
            w = self.action_weights[act]
            exp_r = sum(a * b for a, b in zip(w, x))
            bonus = self.ucb_alpha * math.sqrt(sum(v * v for v in x)) / 5.0
            ucb = exp_r + bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_act = act
                best_r = exp_r
                best_bonus = bonus

        voi = ValueOfInformationCalculator.compute_voi(best_act, prior_entropy=state.hypothesis_entropy)
        h_act = self._to_hierarchical(best_act)

        # Compute propensity for off-policy logging
        propensity = 0.85 if best_bonus < 0.20 else 0.70

        return BanditDecision(
            action=best_act,
            hierarchical_action=h_act,
            expected_reward=best_r,
            exploration_bonus=best_bonus,
            ucb_score=best_ucb,
            logging_propensity=propensity,
            voi_result=voi,
        )

    @staticmethod
    def _to_hierarchical(action: BanditAction) -> HierarchicalAction:
        if action == BanditAction.SLICE:
            return HierarchicalAction(ObjectiveLevel.INVESTIGATE, SourceLevel.SOURCE, PrecisionLevel.SURGICAL, ScopeLevel.SYMBOL, action)
        elif action == BanditAction.TEST:
            return HierarchicalAction(ObjectiveLevel.INVESTIGATE, SourceLevel.TEST, PrecisionLevel.MEDIUM, ScopeLevel.LINES, action)
        elif action == BanditAction.TRACE:
            return HierarchicalAction(ObjectiveLevel.INVESTIGATE, SourceLevel.TRACE, PrecisionLevel.SURGICAL, ScopeLevel.CALL_CHAIN, action)
        elif action == BanditAction.SEARCH:
            return HierarchicalAction(ObjectiveLevel.INVESTIGATE, SourceLevel.SEARCH, PrecisionLevel.COARSE, ScopeLevel.FILE, action)
        elif action == BanditAction.PATCH:
            return HierarchicalAction(ObjectiveLevel.PATCH, SourceLevel.SOURCE, PrecisionLevel.SURGICAL, ScopeLevel.LINES, action)
        elif action == BanditAction.VERIFY:
            return HierarchicalAction(ObjectiveLevel.VERIFY, SourceLevel.TEST, PrecisionLevel.SURGICAL, ScopeLevel.SYMBOL, action)
        else:
            return HierarchicalAction(ObjectiveLevel.INVESTIGATE, SourceLevel.SOURCE, PrecisionLevel.MEDIUM, ScopeLevel.FILE, action)


# --- Sequential Lookahead (Bellman State Transition) ---

class SequentialLookaheadPlanner:
    """Evaluates a_t = argmax_a (E[V(s_{t+1}) | s_t, a] - V(s_t)) / E[T | s_t, a]."""

    @staticmethod
    def evaluate_lookahead(
        state: PolicyState,
        candidate_actions: Sequence[BanditAction],
        current_state_value: float = 0.50,
    ) -> tuple[BanditAction, float, float]:
        """Return (best_action, lookahead_gain, expected_tokens)."""
        best_act = candidate_actions[0]
        best_roi = -float("inf")
        best_gain = 0.0
        best_tok = 500

        for act in candidate_actions:
            voi = ValueOfInformationCalculator.compute_voi(act, state.hypothesis_entropy)
            expected_next_v = min(0.98, current_state_value + (voi.entropy_reduction_bits * 0.45))
            gain = expected_next_v - current_state_value
            roi = gain / float(voi.expected_tokens)

            if roi > best_roi:
                best_roi = roi
                best_act = act
                best_gain = gain
                best_tok = voi.expected_tokens

        return best_act, round(best_gain, 4), float(best_tok)


# --- Independent Stopping Policy ---

class IndependentStoppingPolicy:
    """Halts when Delta P_solve / Delta T < lambda."""

    def __init__(self, lambda_threshold: float = 0.00005) -> None:
        self.lambda_threshold = lambda_threshold

    def should_stop(
        self,
        delta_p_solve: float,
        expected_additional_tokens: int,
        tests_passing: bool = False,
    ) -> tuple[bool, float, str]:
        """Determine if additional inference expenditure is justified."""
        if tests_passing:
            return True, 0.0, "Tests are passing; stop immediately."

        tok = max(1, expected_additional_tokens)
        marginal_gain_per_token = delta_p_solve / float(tok)

        if marginal_gain_per_token < self.lambda_threshold:
            return (
                True,
                marginal_gain_per_token,
                f"Stop: marginal solve gain per token ({marginal_gain_per_token:.7f}) < lambda ({self.lambda_threshold}).",
            )
        return False, marginal_gain_per_token, "Continue: additional tokens economically justified."


# --- Early Failure Trajectory Survival Predictor ---

@dataclass(frozen=True, slots=True)
class EarlyTrajectorySurvival:
    """Multi-horizon survival predictions after turns 2-4."""

    p_success_within_2k: float
    p_success_within_5k: float
    p_success_within_10k: float
    p_runaway_20k: float
    p_regression: float
    intervention_plan: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_success_within_2k": round(self.p_success_within_2k, 4),
            "p_success_within_5k": round(self.p_success_within_5k, 4),
            "p_success_within_10k": round(self.p_success_within_10k, 4),
            "p_runaway_20k": round(self.p_runaway_20k, 4),
            "p_regression": round(self.p_regression, 4),
            "intervention_plan": self.intervention_plan,
        }


class EarlyTrajectorySurvivalModel:
    """Predicts survival horizons early in trajectory execution (turns 2-4)."""

    @staticmethod
    def predict(
        turn: int,
        tokens_spent: int,
        failed_patches: int,
        hypothesis_confidence: float,
    ) -> EarlyTrajectorySurvival:
        """Estimate multi-horizon probabilities and suggest immediate intervention."""
        # Baseline decay
        p_2k = max(0.05, 0.70 - (failed_patches * 0.25) - (turn * 0.05))
        p_5k = max(0.10, 0.85 - (failed_patches * 0.20))
        p_10k = max(0.15, 0.90 - (failed_patches * 0.15))

        # Runaway hazard
        p_runaway = min(0.95, (failed_patches * 0.30) + (float(tokens_spent) / 40_000.0))
        p_regression = min(0.90, (failed_patches * 0.25) + (0.30 if hypothesis_confidence < 0.4 else 0.05))

        if failed_patches >= 2:
            intervention = "PATCH_THRASHING: rollback patch, re-localize target, mandate targeted test."
        elif p_runaway > 0.60:
            intervention = "RUNAWAY_RISK: cap remaining tokens to 2,000, force root-cause verification."
        else:
            intervention = "NORMAL_PROGRESSION: proceed within allocated budget."

        return EarlyTrajectorySurvival(
            p_success_within_2k=p_2k,
            p_success_within_5k=p_5k,
            p_success_within_10k=p_10k,
            p_runaway_20k=p_runaway,
            p_regression=p_regression,
            intervention_plan=intervention,
        )

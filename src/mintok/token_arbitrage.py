"""Token Arbitrage, Marginal Intelligence Escalation, and Dynamic Oracle Economics.

Features:
1. Token Arbitrage Layer:
   - Routes subtasks across heterogeneous model tiers:
     LOCAL_EMBEDDED (redundancy / sanity check) -> CHEAP_API (caller/symbol filtering) -> FRONTIER (code synthesis).
   - Minimizes frontier token usage by sending only the minimum necessary state.
2. Marginal Intelligence Escalation:
   - Evaluates Delta P(solve) / Delta Cost across model tiers.
   - Prevents escalating to expensive frontier models when marginal intelligence gain is below threshold.
3. Dynamic State-Dependent Oracle:
   - Computes state-specific minimum decisive token requirements T_oracle(s).
   - Calculates amplification factor = T_actual / T_oracle(s).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class ModelTier(str, Enum):
    """Execution model tier by capability and cost."""

    LOCAL_EMBEDDED = "LOCAL_EMBEDDED"    # ~0 cost: quantized local coder (e.g. 7B)
    CHEAP_API = "CHEAP_API"              # Low cost: fast API model (e.g. Flash / Lite)
    FRONTIER = "FRONTIER"                # High cost: flagship reasoning model (e.g. Opus / o3)


class SubtaskType(str, Enum):
    """Sub-task category suitable for multi-model delegation."""

    REDUNDANCY_CHECK = "REDUNDANCY_CHECK"
    CALLER_FILTER = "CALLER_FILTER"
    SYNTAX_VALIDATION = "SYNTAX_VALIDATION"
    DIAGNOSTIC_HYPOTHESIS = "DIAGNOSTIC_HYPOTHESIS"
    CODE_SYNTHESIS = "CODE_SYNTHESIS"


@dataclass(frozen=True, slots=True)
class ArbitrageAssignment:
    """Sub-task routing decision with estimated cost."""

    subtask: SubtaskType
    assigned_tier: ModelTier
    estimated_tokens: int
    cost_dollars: float
    frontier_cost_avoided_dollars: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask": self.subtask.value,
            "assigned_tier": self.assigned_tier.value,
            "estimated_tokens": self.estimated_tokens,
            "cost_dollars": round(self.cost_dollars, 5),
            "frontier_cost_avoided_dollars": round(self.frontier_cost_avoided_dollars, 5),
        }


class TokenArbitrageRouter:
    """Assigns sub-tasks to the minimum sufficient model tier."""

    TIER_PRICING: dict[ModelTier, float] = {
        ModelTier.LOCAL_EMBEDDED: 0.00 / 1_000_000.0,
        ModelTier.CHEAP_API: 0.50 / 1_000_000.0,
        ModelTier.FRONTIER: 15.00 / 1_000_000.0,
    }

    @classmethod
    def route_subtask(cls, subtask: SubtaskType, token_budget: int = 1000) -> ArbitrageAssignment:
        """Route subtask to the most cost-effective tier."""
        if subtask in (SubtaskType.REDUNDANCY_CHECK, SubtaskType.SYNTAX_VALIDATION):
            tier = ModelTier.LOCAL_EMBEDDED
        elif subtask in (SubtaskType.CALLER_FILTER, SubtaskType.DIAGNOSTIC_HYPOTHESIS):
            tier = ModelTier.CHEAP_API
        else:
            tier = ModelTier.FRONTIER

        cost = float(token_budget) * cls.TIER_PRICING[tier]
        frontier_cost = float(token_budget) * cls.TIER_PRICING[ModelTier.FRONTIER]
        avoided = max(0.0, frontier_cost - cost)

        return ArbitrageAssignment(
            subtask=subtask,
            assigned_tier=tier,
            estimated_tokens=token_budget,
            cost_dollars=cost,
            frontier_cost_avoided_dollars=avoided,
        )


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Model option with cost and solve profile."""

    model_name: str
    tier: ModelTier
    p_solve: float
    token_cost: int
    dollar_cost: float


@dataclass(frozen=True, slots=True)
class MarginalIntelligenceResult:
    """Outcome of marginal intelligence escalation assessment."""

    selected_model: str
    rejected_escalation: bool
    marginal_gain: float
    marginal_cost_delta: float
    efficiency_ratio: float  # Delta P_solve / Delta Dollar_cost
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "rejected_escalation": self.rejected_escalation,
            "marginal_gain": round(self.marginal_gain, 4),
            "marginal_cost_delta": round(self.marginal_cost_delta, 4),
            "efficiency_ratio": round(self.efficiency_ratio, 2),
            "rationale": self.rationale,
        }


class MarginalIntelligenceEvaluator:
    """Determines whether escalating to higher-tier models is worth the marginal spend."""

    @staticmethod
    def evaluate_escalation(
        current_option: ModelCandidate,
        escalation_option: ModelCandidate,
        efficiency_threshold: float = 1.5,
    ) -> MarginalIntelligenceResult:
        """Compare marginal solve probability gain vs marginal dollar cost."""
        delta_p = escalation_option.p_solve - current_option.p_solve
        delta_cost = escalation_option.dollar_cost - current_option.dollar_cost

        if delta_cost <= 0.0:
            return MarginalIntelligenceResult(
                selected_model=escalation_option.model_name,
                rejected_escalation=False,
                marginal_gain=delta_p,
                marginal_cost_delta=delta_cost,
                efficiency_ratio=100.0,
                rationale="Escalation model is equal or lower cost; upgrade approved.",
            )

        eff_ratio = (delta_p / delta_cost) if delta_cost > 0 else 0.0

        if delta_p <= 0.02 and delta_cost > 0.01:
            return MarginalIntelligenceResult(
                selected_model=current_option.model_name,
                rejected_escalation=True,
                marginal_gain=delta_p,
                marginal_cost_delta=delta_cost,
                efficiency_ratio=eff_ratio,
                rationale=f"Escalation rejected: +{delta_p:.2f} P(solve) is insufficient to justify ${delta_cost:.3f} marginal cost.",
            )

        if eff_ratio < efficiency_threshold:
            return MarginalIntelligenceResult(
                selected_model=current_option.model_name,
                rejected_escalation=True,
                marginal_gain=delta_p,
                marginal_cost_delta=delta_cost,
                efficiency_ratio=eff_ratio,
                rationale=f"Escalation rejected: efficiency ratio {eff_ratio:.2f} is below threshold {efficiency_threshold}.",
            )

        return MarginalIntelligenceResult(
            selected_model=escalation_option.model_name,
            rejected_escalation=False,
            marginal_gain=delta_p,
            marginal_cost_delta=delta_cost,
            efficiency_ratio=eff_ratio,
            rationale="Escalation approved: marginal intelligence gain justifies spend.",
        )


@dataclass(frozen=True, slots=True)
class DynamicOracleCost:
    """Dynamic state-dependent minimum decisive token requirements."""

    task_id: str
    min_evidence_tokens: int
    min_patch_tokens: int
    min_verification_tokens: int
    oracle_total_tokens: int
    actual_tokens_spent: int
    amplification_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "min_evidence_tokens": self.min_evidence_tokens,
            "min_patch_tokens": self.min_patch_tokens,
            "min_verification_tokens": self.min_verification_tokens,
            "oracle_total_tokens": self.oracle_total_tokens,
            "actual_tokens_spent": self.actual_tokens_spent,
            "amplification_factor": round(self.amplification_factor, 2),
        }


class DynamicOracle:
    """Calculates state-specific minimum decisive tokens T_oracle(s)."""

    @staticmethod
    def compute(
        task_id: str,
        actual_tokens_spent: int,
        target_loc: int = 50,
        test_loc: int = 40,
    ) -> DynamicOracleCost:
        """Compute state-dependent oracle and amplification factor."""
        min_ev = int(target_loc * 6)       # ~300 tokens for focused symbol slice
        min_patch = int(target_loc * 3)    # ~150 tokens for minimal diff
        min_verif = int(test_loc * 4)      # ~160 tokens for targeted test
        total_oracle = min_ev + min_patch + min_verif

        amplification = float(actual_tokens_spent) / max(1.0, float(total_oracle))

        return DynamicOracleCost(
            task_id=task_id,
            min_evidence_tokens=min_ev,
            min_patch_tokens=min_patch,
            min_verification_tokens=min_verif,
            oracle_total_tokens=total_oracle,
            actual_tokens_spent=actual_tokens_spent,
            amplification_factor=amplification,
        )

"""Pre-Action Token Budgeting, Progressive Precision Ladder, and Prompt-Cache Economics.

Features:
1. Pre-Action Token Budgeting:
   - Negotiates and enforces strict token envelopes for tool actions (e.g. read_slice with --budget 600).
   - Adapts content granularity (signatures vs slice vs full) to fit the budget.
2. Progressive Precision Ladder:
   - Escalates analysis precision on-demand: AST_LOOKUP -> CALL_GRAPH -> DATA_FLOW -> DYNAMIC_TRACE.
   - Prevents paying for heavyweight dynamic/data-flow analysis when AST lookup suffices.
3. Prompt-Cache Economics:
   - Models tiered pricing: Cost = C_input * T_new + C_cached * T_cached + C_output * T_out.
   - Maximizes prompt-cache prefix stability and scores context locality.
4. Patch Semantic Drift Detector:
   - Detects when agent edits files or functions unrelated to the diagnosed fault location.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class PrecisionLevel(str, Enum):
    """Analysis precision tier in the progressive escalation ladder."""

    AST_LOOKUP = "AST_LOOKUP"          # ~100 tokens: signatures, docstrings, symbols
    CALL_GRAPH = "CALL_GRAPH"          # ~400 tokens: callers, callees, dependency edges
    DATA_FLOW = "DATA_FLOW"            # ~1200 tokens: variable def-use chains, slice body
    DYNAMIC_TRACE = "DYNAMIC_TRACE"    # ~3000 tokens: test execution trace, variable states


@dataclass(frozen=True, slots=True)
class BudgetedActionSpec:
    """Action specification adapted to fit a negotiated token budget."""

    action_type: str
    target: str
    requested_budget: int
    precision_level: PrecisionLevel
    estimated_tokens: int
    content_format: str  # "signatures_only" | "compact_slice" | "full_slice"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "requested_budget": self.requested_budget,
            "precision_level": self.precision_level.value,
            "estimated_tokens": self.estimated_tokens,
            "content_format": self.content_format,
        }


class ActionBudgetNegotiator:
    """Enforces pre-action token budgets and adapts content granularity."""

    @staticmethod
    def negotiate_budget(
        action_type: str,
        target: str,
        budget: int = 800,
    ) -> BudgetedActionSpec:
        """Configure tool action parameters to strictly respect budget envelope."""
        b = max(100, budget)

        if b < 300:
            return BudgetedActionSpec(
                action_type=action_type,
                target=target,
                requested_budget=b,
                precision_level=PrecisionLevel.AST_LOOKUP,
                estimated_tokens=min(b, 200),
                content_format="signatures_only",
            )
        elif b < 1000:
            return BudgetedActionSpec(
                action_type=action_type,
                target=target,
                requested_budget=b,
                precision_level=PrecisionLevel.CALL_GRAPH,
                estimated_tokens=min(b, 650),
                content_format="compact_slice",
            )
        else:
            return BudgetedActionSpec(
                action_type=action_type,
                target=target,
                requested_budget=b,
                precision_level=PrecisionLevel.DATA_FLOW,
                estimated_tokens=min(b, 1500),
                content_format="full_slice",
            )


class ProgressivePrecisionCoordinator:
    """Selects minimum sufficient precision level based on residual uncertainty."""

    @staticmethod
    def select_precision(
        residual_uncertainty: float,
        previous_failed_attempts: int = 0,
    ) -> PrecisionLevel:
        """Escalate precision only as warranted by uncertainty and failed attempts."""
        if previous_failed_attempts >= 2 or residual_uncertainty > 0.85:
            return PrecisionLevel.DYNAMIC_TRACE
        if residual_uncertainty > 0.60 or previous_failed_attempts >= 1:
            return PrecisionLevel.DATA_FLOW
        if residual_uncertainty > 0.30:
            return PrecisionLevel.CALL_GRAPH
        return PrecisionLevel.AST_LOOKUP


@dataclass(frozen=True, slots=True)
class PromptCacheCost:
    """Cost breakdown considering prompt cache hits and writes."""

    new_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_dollars: float
    cache_hit_ratio: float
    locality_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_tokens": self.new_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "cost_dollars": round(self.cost_dollars, 6),
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "locality_score": round(self.locality_score, 4),
        }


class PromptCacheEconomics:
    """Calculates cache-aware token economics and optimizes context prefix locality."""

    def __init__(
        self,
        c_input: float = 3.00 / 1_000_000.0,      # $3.00 / 1M tokens new input
        c_cached: float = 0.30 / 1_000_000.0,    # $0.30 / 1M tokens cached (10% cost)
        c_output: float = 15.00 / 1_000_000.0,   # $15.00 / 1M tokens output
    ) -> None:
        self.c_input = c_input
        self.c_cached = c_cached
        self.c_output = c_output

    def compute_cost(
        self,
        new_tokens: int,
        cached_tokens: int,
        output_tokens: int,
    ) -> PromptCacheCost:
        """Compute dollar cost under prompt caching."""
        total_input = new_tokens + cached_tokens
        hit_ratio = float(cached_tokens) / max(1.0, float(total_input))
        cost = (
            (float(new_tokens) * self.c_input)
            + (float(cached_tokens) * self.c_cached)
            + (float(output_tokens) * self.c_output)
        )
        locality = hit_ratio

        return PromptCacheCost(
            new_tokens=new_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            cost_dollars=cost,
            cache_hit_ratio=hit_ratio,
            locality_score=locality,
        )

    def evaluate_append_vs_rewrite(
        self,
        existing_cached_tokens: int,
        appended_tokens: int,
        output_tokens: int = 500,
    ) -> tuple[PromptCacheCost, PromptCacheCost]:
        """Compare appending observations to prefix vs rewriting/compacting prefix."""
        # Append preserves cache prefix
        append_cost = self.compute_cost(
            new_tokens=appended_tokens,
            cached_tokens=existing_cached_tokens,
            output_tokens=output_tokens,
        )
        # Rewrite busts cache prefix: all existing tokens are re-ingested as new
        rewrite_cost = self.compute_cost(
            new_tokens=existing_cached_tokens + appended_tokens,
            cached_tokens=0,
            output_tokens=output_tokens,
        )
        return append_cost, rewrite_cost


class SemanticDriftDetector:
    """Detects when proposed patches drift away from diagnosed fault targets."""

    @staticmethod
    def evaluate_drift(
        target_file: str,
        target_symbols: Sequence[str],
        patch_file: str,
        patch_symbols: Sequence[str],
        patch_line_distance: int = 0,
    ) -> tuple[float, str]:
        """Compute drift score (0.0 = aligned, 1.0 = severe drift) and rationale."""
        if target_file and patch_file and target_file != patch_file:
            return 0.85, f"Patch touches file '{patch_file}' but fault is localized to '{target_file}'"

        if target_symbols and patch_symbols:
            common = set(target_symbols).intersection(set(patch_symbols))
            if not common:
                return 0.65, f"Patch modifies {list(patch_symbols)} which does not overlap target symbol {list(target_symbols)}"

        if patch_line_distance > 100:
            return 0.40, f"Patch line distance {patch_line_distance} is far from identified fault site"

        return 0.05, "Patch is tightly aligned with diagnosed fault site"

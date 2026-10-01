"""Patch Risk, Reversibility, Minimum Sufficient Patch Predictor, and Avoidable Spend Breakdown.

Features:
1. Minimum Sufficient Patch Predictor:
   - Predicts expected patch size (files, lines, symbols).
   - Detects hypothesis loss: flags when actual patch bloats significantly over prediction (e.g. 1 file/8 lines vs 4 files/180 lines).
2. Patch Reversibility Optimization:
   - Scores reversibility and blast radius: allows cheap reversible patches immediately,
     while gating irreversible multi-file refactors behind confirming observations.
3. Inference Milestone Tracking:
   - Tracks tokens to first correct hypothesis and tokens to irreversible mistake.
4. Avoidable Spend Accounting:
   - Disaggregates waste into: necessary, unavoidable uncertainty, avoidable redundancy,
     avoidable recovery, avoidable verification, and catastrophic waste.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class ReversibilityTier(str, Enum):
    """Reversibility classification of a code change."""

    REVERSIBLE_LOCAL = "REVERSIBLE_LOCAL"      # Single function or file, trivial git restore
    MEDIUM_RISK = "MEDIUM_RISK"                # Multi-function or cross-file edit
    IRREVERSIBLE_HIGH_RISK = "HIGH_RISK"       # Wide refactor, schema change, destructive deletion


@dataclass(frozen=True, slots=True)
class PatchExpectation:
    """Predicted minimum sufficient patch dimensions."""

    expected_files: int
    expected_lines: int
    expected_symbols: int
    tolerance_ratio: float = 2.5  # Allowed bloat factor before triggering hypothesis loss alarm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PatchAnomalyVerdict:
    """Assessment of whether an actual patch conforms to minimum sufficient expectation."""

    is_anomaly: bool
    predicted_lines: int
    actual_lines: int
    bloat_ratio: float
    recommendation: str  # "PROCEED" | "REASSESS_HYPOTHESIS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "predicted_lines": self.predicted_lines,
            "actual_lines": self.actual_lines,
            "bloat_ratio": round(self.bloat_ratio, 2),
            "recommendation": self.recommendation,
        }


class MinimumSufficientPatchPredictor:
    """Predicts required patch footprint and detects runaway patch bloat."""

    @staticmethod
    def predict_expectation(
        task_family: str,
        target_loc: int = 50,
    ) -> PatchExpectation:
        """Estimate required patch dimensions from task profile."""
        tf = task_family.lower()
        if "bugfix" in tf or "small" in tf or "local" in tf:
            return PatchExpectation(expected_files=1, expected_lines=8, expected_symbols=1)
        elif "test" in tf:
            return PatchExpectation(expected_files=2, expected_lines=18, expected_symbols=2)
        elif "schema" in tf or "api" in tf:
            return PatchExpectation(expected_files=3, expected_lines=35, expected_symbols=4)
        else:
            return PatchExpectation(expected_files=1, expected_lines=15, expected_symbols=2)

    @classmethod
    def evaluate_proposed_patch(
        cls,
        expectation: PatchExpectation,
        actual_files_count: int,
        actual_lines_count: int,
    ) -> PatchAnomalyVerdict:
        """Check if proposed edit exceeds minimum sufficient patch envelope."""
        bloat = float(actual_lines_count) / max(1.0, float(expectation.expected_lines))
        is_anomaly = (
            actual_files_count > (expectation.expected_files * 2)
            or bloat > expectation.tolerance_ratio
        )
        rec = "REASSESS_HYPOTHESIS" if is_anomaly else "PROCEED"

        return PatchAnomalyVerdict(
            is_anomaly=is_anomaly,
            predicted_lines=expectation.expected_lines,
            actual_lines=actual_lines_count,
            bloat_ratio=bloat,
            recommendation=rec,
        )


class PatchReversibilityEvaluator:
    """Evaluates patch reversibility and gates risky modifications."""

    @staticmethod
    def classify_reversibility(
        files_modified_count: int,
        lines_modified_count: int,
        deletions_count: int,
        touches_database_or_schema: bool = False,
    ) -> tuple[ReversibilityTier, str]:
        """Classify change reversibility and provide policy decision."""
        if touches_database_or_schema or deletions_count > 100 or files_modified_count > 4:
            return (
                ReversibilityTier.IRREVERSIBLE_HIGH_RISK,
                "High-risk irreversible modification detected. Require confirming observation and targeted test before applying.",
            )
        elif files_modified_count > 1 or lines_modified_count > 40:
            return (
                ReversibilityTier.MEDIUM_RISK,
                "Medium-risk change: localized within multiple files. Run verification immediately after edit.",
            )
        return (
            ReversibilityTier.REVERSIBLE_LOCAL,
            "Low-risk localized edit: completely reversible with single git checkout.",
        )


@dataclass(frozen=True, slots=True)
class AvoidableTokensBreakdown:
    """Granular categorization of trajectory token spend."""

    total_tokens: int
    necessary_tokens: int
    unavoidable_uncertainty_tokens: int
    avoidable_redundancy_tokens: int
    avoidable_recovery_tokens: int
    avoidable_verification_tokens: int
    catastrophic_waste_tokens: int
    total_avoidable_tokens: int
    avoidable_eliminated_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "necessary_tokens": self.necessary_tokens,
            "unavoidable_uncertainty_tokens": self.unavoidable_uncertainty_tokens,
            "avoidable_redundancy_tokens": self.avoidable_redundancy_tokens,
            "avoidable_recovery_tokens": self.avoidable_recovery_tokens,
            "avoidable_verification_tokens": self.avoidable_verification_tokens,
            "catastrophic_waste_tokens": self.catastrophic_waste_tokens,
            "total_avoidable_tokens": self.total_avoidable_tokens,
            "avoidable_eliminated_ratio": round(self.avoidable_eliminated_ratio, 4),
        }


class InferenceMilestoneTracker:
    """Tracks token efficiency milestones throughout trajectory progression."""

    def __init__(self) -> None:
        self.tokens_to_first_correct_hypothesis: int | None = None
        self.tokens_to_first_destructive_action: int | None = None
        self.total_tokens_spent: int = 0
        self.correct_hypothesis_found: bool = False
        self.destructive_action_occurred: bool = False

    def record_step(
        self,
        step_tokens: int,
        is_correct_hypothesis: bool = False,
        is_destructive_action: bool = False,
    ) -> None:
        """Update milestones given turn outcome."""
        self.total_tokens_spent += step_tokens

        if is_correct_hypothesis and not self.correct_hypothesis_found:
            self.correct_hypothesis_found = True
            self.tokens_to_first_correct_hypothesis = self.total_tokens_spent

        if is_destructive_action and not self.destructive_action_occurred:
            self.destructive_action_occurred = True
            self.tokens_to_first_destructive_action = self.total_tokens_spent

    def compute_avoidable_breakdown(
        self,
        oracle_tokens: int = 5000,
        redundant_reads_count: int = 0,
        failed_patches_count: int = 0,
        excess_verification_tokens: int = 0,
    ) -> AvoidableTokensBreakdown:
        """Calculate detailed breakdown of avoidable vs necessary token expenditure."""
        necessary = min(self.total_tokens_spent, oracle_tokens)
        unavoidable = int(necessary * 0.20)
        redundancy = redundant_reads_count * 2500
        recovery = failed_patches_count * 3500
        verif = excess_verification_tokens
        catastrophic = max(0, self.total_tokens_spent - (necessary + unavoidable + redundancy + recovery + verif))
        avoidable_total = redundancy + recovery + verif + catastrophic

        eliminated_ratio = 1.0 - (float(avoidable_total) / max(1.0, float(self.total_tokens_spent)))

        return AvoidableTokensBreakdown(
            total_tokens=self.total_tokens_spent,
            necessary_tokens=necessary,
            unavoidable_uncertainty_tokens=unavoidable,
            avoidable_redundancy_tokens=redundancy,
            avoidable_recovery_tokens=recovery,
            avoidable_verification_tokens=verif,
            catastrophic_waste_tokens=catastrophic,
            total_avoidable_tokens=avoidable_total,
            avoidable_eliminated_ratio=max(0.0, min(1.0, eliminated_ratio)),
        )

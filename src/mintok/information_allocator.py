"""Information Density Allocator, Evidence Sufficiency, and Adaptive Verification.

Features:
1. Evidence Sufficiency Model (5 Gates):
   - TARGET_KNOWN, CALLER_KNOWN, FAILURE_KNOWN, PATCH_LOCATION_KNOWN, EXPECTED_BEHAVIOR_KNOWN.
   - When all 5 are known, locks further reads and directs agent immediately to patch.
2. Information-Density Scorer:
   - Evaluates action by expected_uncertainty_reduction / expected_tokens.
3. Adaptive Verifier:
   - Estimates blast radius (LOCAL_FUNCTION, PUBLIC_API, CROSS_PACKAGE, HIGH_RISK)
     to run the smallest decisive test suite.
   - Tracks verification tokens per correctly verified patch.
4. Local Action Critic:
   - Classifies proposed actions before frontier generation into:
     LIKELY_REDUNDANT, LIKELY_USEFUL, HIGH_RISK, NEEDS_VERIFICATION.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class BlastRadius(str, Enum):
    """Estimated impact radius of a proposed code change."""

    LOCAL_FUNCTION = "LOCAL_FUNCTION"
    PUBLIC_API = "PUBLIC_API"
    CROSS_PACKAGE = "CROSS_PACKAGE"
    HIGH_RISK = "HIGH_RISK"


class CriticClassification(str, Enum):
    """Pre-generation action classification by local critic."""

    LIKELY_REDUNDANT = "LIKELY_REDUNDANT"
    LIKELY_USEFUL = "LIKELY_USEFUL"
    HIGH_RISK = "HIGH_RISK"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"


@dataclass
class EvidenceSufficiencyState:
    """Status of the 5 requisite gates before code modification."""

    target_known: bool = False
    caller_known: bool = False
    failure_known: bool = False
    patch_location_known: bool = False
    expected_behavior_known: bool = False

    @property
    def is_sufficient_to_patch(self) -> bool:
        """All 5 gates are satisfied; further contextual reads should be avoided."""
        return (
            self.target_known
            and self.caller_known
            and self.failure_known
            and self.patch_location_known
            and self.expected_behavior_known
        )

    def missing_gates(self) -> list[str]:
        missing = []
        if not self.target_known:
            missing.append("TARGET_KNOWN")
        if not self.caller_known:
            missing.append("CALLER_KNOWN")
        if not self.failure_known:
            missing.append("FAILURE_KNOWN")
        if not self.patch_location_known:
            missing.append("PATCH_LOCATION_KNOWN")
        if not self.expected_behavior_known:
            missing.append("EXPECTED_BEHAVIOR_KNOWN")
        return missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_known": self.target_known,
            "caller_known": self.caller_known,
            "failure_known": self.failure_known,
            "patch_location_known": self.patch_location_known,
            "expected_behavior_known": self.expected_behavior_known,
            "is_sufficient_to_patch": self.is_sufficient_to_patch,
            "missing_gates": self.missing_gates(),
        }


@dataclass(frozen=True, slots=True)
class ActionInformationDensity:
    """Action scored by uncertainty reduction per token consumed."""

    action_type: str
    target: str
    hypotheses_eliminated: int
    expected_tokens: int
    information_density: float  # hypotheses_eliminated / expected_tokens

    @classmethod
    def compute(
        cls,
        action_type: str,
        target: str,
        hypotheses_eliminated: int,
        expected_tokens: int,
    ) -> ActionInformationDensity:
        tokens = max(1, expected_tokens)
        density = float(hypotheses_eliminated) / float(tokens)
        return cls(
            action_type=action_type,
            target=target,
            hypotheses_eliminated=hypotheses_eliminated,
            expected_tokens=expected_tokens,
            information_density=density,
        )


class InformationDensityScorer:
    """Ranks proposed actions by expected uncertainty reduction per token."""

    @staticmethod
    def score_actions(
        candidate_actions: Sequence[dict[str, Any]],
    ) -> list[ActionInformationDensity]:
        scores = []
        for a in candidate_actions:
            act_type = str(a.get("action_type", ""))
            target = str(a.get("target", ""))
            hyp = int(a.get("hypotheses_eliminated", 1))
            tokens = int(a.get("expected_tokens", 1000))
            scores.append(
                ActionInformationDensity.compute(
                    action_type=act_type,
                    target=target,
                    hypotheses_eliminated=hyp,
                    expected_tokens=tokens,
                )
            )
        # Descending sort by information density
        return sorted(scores, key=lambda s: s.information_density, reverse=True)


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """Selected verification scope and estimated token spend."""

    blast_radius: BlastRadius
    recommended_scope: str  # "unit_test" | "affected_tests" | "package_suite" | "full_suite"
    estimated_tokens: int
    rationale: str


class AdaptiveVerifier:
    """Determines the smallest decisive verification test based on estimated change blast radius."""

    def __init__(self) -> None:
        self.total_verification_tokens: int = 0
        self.verified_patches_count: int = 0

    def estimate_blast_radius(
        self,
        files_modified: Sequence[str],
        symbols_modified: Sequence[str],
        is_public_api_modified: bool = False,
        cross_package: bool = False,
        schema_modified: bool = False,
    ) -> BlastRadius:
        """Classify change blast radius."""
        if schema_modified or len(files_modified) > 4:
            return BlastRadius.HIGH_RISK
        if cross_package:
            return BlastRadius.CROSS_PACKAGE
        if is_public_api_modified:
            return BlastRadius.PUBLIC_API
        return BlastRadius.LOCAL_FUNCTION

    def select_verification(
        self,
        blast_radius: BlastRadius,
        target_unit_test: str = "tests/test_unit.py",
        affected_test_file: str = "tests/test_api.py",
    ) -> VerificationDecision:
        """Select minimal decisive test suite based on blast radius."""
        if blast_radius == BlastRadius.LOCAL_FUNCTION:
            return VerificationDecision(
                blast_radius=blast_radius,
                recommended_scope="unit_test",
                estimated_tokens=450,
                rationale=f"Local function modification: run minimal targeted unit test ({target_unit_test})",
            )
        if blast_radius == BlastRadius.PUBLIC_API:
            return VerificationDecision(
                blast_radius=blast_radius,
                recommended_scope="affected_tests",
                estimated_tokens=1800,
                rationale=f"Public API signature change: run caller affected tests ({affected_test_file})",
            )
        if blast_radius == BlastRadius.CROSS_PACKAGE:
            return VerificationDecision(
                blast_radius=blast_radius,
                recommended_scope="package_suite",
                estimated_tokens=4200,
                rationale="Cross-package dependency change: run package integration suite",
            )
        return VerificationDecision(
            blast_radius=blast_radius,
            recommended_scope="full_suite",
            estimated_tokens=14000,
            rationale="High-risk schema or multi-module refactor: full regression suite required",
        )

    def record_verification_result(self, tokens_spent: int, success: bool) -> None:
        self.total_verification_tokens += tokens_spent
        if success:
            self.verified_patches_count += 1

    @property
    def verification_tokens_per_verified_patch(self) -> float:
        return float(self.total_verification_tokens) / max(1.0, float(self.verified_patches_count))


class LocalActionCritic:
    """Pre-generation inspector evaluating proposed action utility and risk."""

    def evaluate(
        self,
        proposed_action: str,
        target: str,
        known_read_targets: Sequence[str],
        has_pending_patch: bool,
        sufficiency: EvidenceSufficiencyState | None = None,
    ) -> tuple[CriticClassification, str]:
        """Classify proposed action into LIKELY_REDUNDANT, LIKELY_USEFUL, HIGH_RISK, NEEDS_VERIFICATION."""
        act_lower = proposed_action.lower()

        # Check for unverified patch followed by another patch
        if any(k in act_lower for k in ("patch", "apply_patch", "edit")) and has_pending_patch:
            return (
                CriticClassification.NEEDS_VERIFICATION,
                "A prior patch has not been verified yet. Run targeted verifier before applying another edit.",
            )

        # Check for redundant reads when evidence is already sufficient
        if any(k in act_lower for k in ("read", "slice", "cat", "view")):
            if sufficiency and sufficiency.is_sufficient_to_patch:
                return (
                    CriticClassification.LIKELY_REDUNDANT,
                    "Evidence sufficiency gates are already satisfied (target, caller, failure, location known). Proceed to patch rather than consuming additional read tokens.",
                )
            if target in known_read_targets:
                return (
                    CriticClassification.LIKELY_REDUNDANT,
                    f"Target '{target}' has already been ingested in recent context.",
                )

        # High-risk wide edit detection
        if "delete" in act_lower or "drop" in act_lower:
            return (
                CriticClassification.HIGH_RISK,
                "Destructive delete action proposed; verify against AST symbol preservation.",
            )

        return CriticClassification.LIKELY_USEFUL, "Action evaluated as productive."

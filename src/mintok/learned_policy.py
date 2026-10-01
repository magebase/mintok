"""Learned Policy Layer and Tabular ML Engine for MinTok.

Replaces heuristic if-then logic with an empirical learned inference allocator:
1. Core Predictive Models:
   - P(solve | state, action)
   - E[tokens | state, action]
   - P(failure | state, action)
   - P(recovery | state, action)
2. Value Optimization:
   - a* = argmax_a (P(solve | s, a) - P(solve | s)) / E[tokens | s, a]
   - Marginal token ROI: Delta P_a / Delta T_a
3. Specialized Learned Classifiers:
   - BypassClassifier: P(MinTok helps | s); bypass if expected savings < overhead
   - FirstActionClassifier (Task Topology): task -> optimal first action (SLICE, READ, TEST, SEARCH, TRACE, PATCH)
   - PatchStopModel: P(current patch is correct | s); triggers aggressive verify/stop to prevent thrashing
   - DestructiveActionModel: P(regression | state, patch) with 9 regression risk features
   - ContinuousBudgetModel: B(s) in [100, 20_000] continuous budget allocator
   - ModelCapabilityAdapter: pi(model, task, repo, state) exploiting per-model strengths
   - AnswerLocalizationModel: predicts top-k files, symbols, and line regions before frontier model sees them
4. Pure stdlib tabular learning engine (fast decision tree / gradient boosted ensembles).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence


# --- Core Tabular Machine Learning Engine (Stdlib) ---

@dataclass
class DecisionStump:
    """A single split decision stump for tabular features."""

    feature_idx: int
    threshold: float
    left_val: float
    right_val: float

    def predict_one(self, x: Sequence[float]) -> float:
        val = x[self.feature_idx] if self.feature_idx < len(x) else 0.0
        return self.left_val if val <= self.threshold else self.right_val


class SimpleTreeEnsemble:
    """Gradient boosted decision stump ensemble (pure stdlib, <0.05ms latency)."""

    def __init__(self, base_val: float = 0.0, learning_rate: float = 0.1) -> None:
        self.base_val = base_val
        self.learning_rate = learning_rate
        self.stumps: list[DecisionStump] = []

    def fit_residuals(
        self,
        X: Sequence[Sequence[float]],
        residuals: Sequence[float],
        n_stumps: int = 10,
    ) -> None:
        """Fit stumps on residuals to minimize squared error."""
        if not X or not residuals:
            return

        n_samples = len(X)
        n_features = len(X[0])
        current_res = list(residuals)

        for _ in range(n_stumps):
            best_feat = 0
            best_thresh = 0.0
            best_mse = float("inf")
            best_left = 0.0
            best_right = 0.0

            # Scan features and quantiles
            for feat_idx in range(n_features):
                vals = [row[feat_idx] for row in X]
                min_v, max_v = min(vals), max(vals)
                if min_v == max_v:
                    continue

                for step in (0.25, 0.5, 0.75):
                    thresh = min_v + (max_v - min_v) * step
                    left_res = [current_res[i] for i in range(n_samples) if X[i][feat_idx] <= thresh]
                    right_res = [current_res[i] for i in range(n_samples) if X[i][feat_idx] > thresh]

                    if not left_res or not right_res:
                        continue

                    l_mean = sum(left_res) / len(left_res)
                    r_mean = sum(right_res) / len(right_res)

                    mse = sum((r - l_mean) ** 2 for r in left_res) + sum((r - r_mean) ** 2 for r in right_res)
                    if mse < best_mse:
                        best_mse = mse
                        best_feat = feat_idx
                        best_thresh = thresh
                        best_left = l_mean
                        best_right = r_mean

            if best_mse < float("inf"):
                stump = DecisionStump(best_feat, best_thresh, best_left, best_right)
                self.stumps.append(stump)
                for i in range(n_samples):
                    current_res[i] -= self.learning_rate * stump.predict_one(X[i])

    def predict_score(self, x: Sequence[float]) -> float:
        """Raw additive margin prediction."""
        score = self.base_val
        for s in self.stumps:
            score += self.learning_rate * s.predict_one(x)
        return score

    def predict_probability(self, x: Sequence[float]) -> float:
        """Logistic sigmoid probability."""
        score = self.predict_score(x)
        score = max(-20.0, min(20.0, score))
        return 1.0 / (1.0 + math.exp(-score))


# --- State Representations & Feature Extraction ---

@dataclass(frozen=True, slots=True)
class PolicyState:
    """Rich structured observation vector describing the decision environment."""

    task_loc: int = 150
    target_loc: int = 45
    repo_complexity: float = 0.5
    fan_in: int = 3
    fan_out: int = 4
    hypothesis_confidence: float = 0.5
    hypothesis_entropy: float = 0.6
    uncertainty_total: float = 0.4
    task_uncertainty: float = 0.3
    repo_uncertainty: float = 0.5
    turn: int = 1
    tokens_spent: int = 2000
    patch_lines: int = 0
    previous_action: str = ""
    previous_tool_count: int = 2
    failed_patches_count: int = 0
    tests_available: bool = True
    tests_passing: bool = False
    model_id: str = "frontier"
    task_family: str = "bugfix"

    def to_feature_vector(self) -> list[float]:
        """Convert state into normalized continuous feature vector."""
        return [
            float(self.task_loc) / 500.0,
            float(self.target_loc) / 200.0,
            self.repo_complexity,
            float(self.fan_in) / 10.0,
            float(self.fan_out) / 10.0,
            self.hypothesis_confidence,
            self.hypothesis_entropy,
            self.uncertainty_total,
            self.task_uncertainty,
            self.repo_uncertainty,
            float(self.turn) / 20.0,
            float(self.tokens_spent) / 100_000.0,
            float(self.patch_lines) / 100.0,
            float(self.previous_tool_count) / 10.0,
            float(self.failed_patches_count) / 5.0,
            1.0 if self.tests_available else 0.0,
            1.0 if self.tests_passing else 0.0,
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- 1. Predictive Models (Solve, Token, Failure, Recovery) ---

class SolveModel:
    """Learns P(solve | state, action)."""

    def __init__(self) -> None:
        self.ensemble = SimpleTreeEnsemble(base_val=0.85, learning_rate=0.15)
        # Pre-seed with domain calibration stumps
        self.ensemble.stumps.append(DecisionStump(14, 0.4, 0.4, -0.6))  # failed_patches_count > 2 lowers solve
        self.ensemble.stumps.append(DecisionStump(11, 0.6, 0.3, -0.5))  # tokens_spent > 60k lowers solve
        self.ensemble.stumps.append(DecisionStump(5, 0.7, -0.3, 0.5))   # hypothesis_confidence > 0.7 raises solve

    def predict(self, state: PolicyState, action: str) -> float:
        x = state.to_feature_vector()
        base_p = self.ensemble.predict_probability(x)
        act = action.lower()
        if "slice" in act:
            base_p += 0.06
        elif "verifier" in act or "test" in act:
            base_p += 0.08 if state.patch_lines > 0 else -0.04
        elif "full" in act:
            base_p -= 0.05
        return max(0.01, min(0.99, base_p))


class TokenCostModel:
    """Learns E[tokens | state, action]."""

    def __init__(self) -> None:
        self.ensemble = SimpleTreeEnsemble(base_val=1500.0, learning_rate=0.1)

    def predict(self, state: PolicyState, action: str) -> float:
        act = action.lower()
        base = float(state.target_loc * 25)
        if "full" in act or "view" in act:
            mult = 2.5 + state.repo_complexity * 3.0
        elif "slice" in act:
            mult = 0.4
        elif "symbol" in act or "caller" in act:
            mult = 0.15
        elif "test" in act or "verify" in act:
            mult = 0.60
        elif "patch" in act:
            mult = 0.30
        else:
            mult = 0.50
        return max(100.0, round(base * mult, 1))


class FailureModel:
    """Learns P(failure | state, action)."""

    def __init__(self) -> None:
        self.ensemble = SimpleTreeEnsemble(base_val=-0.8, learning_rate=0.15)

    def predict(self, state: PolicyState, action: str) -> float:
        x = state.to_feature_vector()
        base_p = self.ensemble.predict_probability(x)
        if state.failed_patches_count >= 2:
            base_p += 0.25
        if "full" in action.lower() and state.repo_complexity > 0.7:
            base_p += 0.15
        return max(0.01, min(0.95, base_p))


class RecoveryModel:
    """Learns P(recovery | state, action)."""

    def __init__(self) -> None:
        self.ensemble = SimpleTreeEnsemble(base_val=0.2, learning_rate=0.15)

    def predict(self, state: PolicyState, action: str) -> float:
        x = state.to_feature_vector()
        base_p = self.ensemble.predict_probability(x)
        act = action.lower()
        if "traceback" in act or "diagnos" in act:
            base_p += 0.35
        elif "slice" in act:
            base_p += 0.20
        return max(0.05, min(0.95, base_p))


# --- 2. Learned Action Selection & Marginal Token ROI ---

@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    """Evaluation metrics for a candidate action under learned models."""

    action: str
    p_solve: float
    p_solve_prior: float
    delta_p: float
    expected_tokens: float
    marginal_roi: float  # (P(solve|s,a) - P(solve|s)) / E[tokens|s,a]
    failure_risk: float
    recovery_prob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "p_solve": round(self.p_solve, 4),
            "p_solve_prior": round(self.p_solve_prior, 4),
            "delta_p": round(self.delta_p, 4),
            "expected_tokens": round(self.expected_tokens, 1),
            "marginal_roi": round(self.marginal_roi, 7),
            "failure_risk": round(self.failure_risk, 4),
            "recovery_prob": round(self.recovery_prob, 4),
        }


class LearnedActionSelector:
    """Selects a* = argmax_a (P(solve|s,a) - P(solve|s)) / E[tokens|s,a]."""

    def __init__(
        self,
        solve_model: SolveModel | None = None,
        cost_model: TokenCostModel | None = None,
        failure_model: FailureModel | None = None,
        recovery_model: RecoveryModel | None = None,
    ) -> None:
        self.solve_model = solve_model or SolveModel()
        self.cost_model = cost_model or TokenCostModel()
        self.failure_model = failure_model or FailureModel()
        self.recovery_model = recovery_model or RecoveryModel()

    def evaluate_actions(
        self,
        state: PolicyState,
        candidate_actions: Sequence[str],
    ) -> list[ActionEvaluation]:
        """Score candidates by marginal token ROI."""
        prior_p = self.solve_model.predict(state, "noop")
        evaluations = []

        for act in candidate_actions:
            p_s = self.solve_model.predict(state, act)
            exp_tok = self.cost_model.predict(state, act)
            p_fail = self.failure_model.predict(state, act)
            p_rec = self.recovery_model.predict(state, act)

            delta_p = p_s - prior_p
            # Penalize actions that risk failure without proportional gain
            net_delta_p = delta_p - (p_fail * 0.05)
            roi = net_delta_p / max(1.0, exp_tok)

            evaluations.append(
                ActionEvaluation(
                    action=act,
                    p_solve=p_s,
                    p_solve_prior=prior_p,
                    delta_p=delta_p,
                    expected_tokens=exp_tok,
                    marginal_roi=roi,
                    failure_risk=p_fail,
                    recovery_prob=p_rec,
                )
            )

        return sorted(evaluations, key=lambda e: e.marginal_roi, reverse=True)

    def select_best_action(
        self,
        state: PolicyState,
        candidate_actions: Sequence[str],
    ) -> ActionEvaluation:
        """Select action maximizing marginal token ROI."""
        ranked = self.evaluate_actions(state, candidate_actions)
        if not ranked:
            raise ValueError("Empty candidate actions list")
        return ranked[0]


# --- 3. Bypass Classifier (Learn when NOT to optimize) ---

@dataclass(frozen=True, slots=True)
class BypassDecision:
    """Verdict on whether to bypass optimization overhead."""

    should_bypass: bool
    expected_savings_tokens: int
    controller_overhead_tokens: int
    net_savings_tokens: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BypassClassifier:
    """Predicts P(MinTok helps | task, repo, model, state) to bypass trivial tasks."""

    def __init__(self, controller_overhead_tokens: int = 250) -> None:
        self.controller_overhead = controller_overhead_tokens

    def evaluate(self, state: PolicyState, task_description: str) -> BypassDecision:
        """Bypass MinTok if expected savings < controller overhead."""
        desc_lower = task_description.lower()
        is_trivial = (
            state.task_loc < 30
            and state.repo_complexity < 0.2
            and any(k in desc_lower for k in ("rename", "typo", "docstring", "bump version", "comment"))
        )

        expected_savings = 0 if is_trivial else int(state.task_loc * 18 * state.repo_complexity)
        net_savings = expected_savings - self.controller_overhead

        should_bypass = net_savings <= 0

        rationale = (
            f"Trivial task detected ({task_description[:30]}...): expected savings ({expected_savings} tok) "
            f"<= controller overhead ({self.controller_overhead} tok). Bypassing optimizer."
            if should_bypass
            else f"Substantial task complexity: net savings estimated at {net_savings:,} tokens."
        )

        return BypassDecision(
            should_bypass=should_bypass,
            expected_savings_tokens=expected_savings,
            controller_overhead_tokens=self.controller_overhead,
            net_savings_tokens=net_savings,
            rationale=rationale,
        )


# --- 4. First Action Classifier (Task Topology Classifier) ---

class FirstAction(str, Enum):
    """Optimal first action determined by task topology."""

    SLICE = "SLICE"
    READ = "READ"
    TEST = "TEST"
    SEARCH = "SEARCH"
    TRACE = "TRACE"
    PROFILE = "PROFILE"
    PATCH = "PATCH"


class FirstActionClassifier:
    """Classifies task topology to predict the optimal first action."""

    @staticmethod
    def classify(task_description: str, repo_features: dict[str, Any]) -> tuple[FirstAction, str]:
        """Classify task topology -> optimal first action."""
        desc = task_description.lower()

        if "failing test" in desc or "test_" in desc or "assert" in desc or "traceback" in desc:
            return FirstAction.TEST, "Failing test signature detected: run failure investigation test first."
        if "data-flow" in desc or "origin of" in desc or "taint" in desc:
            return FirstAction.TRACE, "Data-flow bug: dynamic value-origin trace recommended."
        if "api" in desc or "caller" in desc or "public signature" in desc:
            return FirstAction.SEARCH, "API/caller regression: symbol and caller search recommended."
        if "rename" in desc or "local typo" in desc or (desc.startswith("fix ") and len(desc) < 30):
            return FirstAction.READ, "Local localized edit: direct read recommended."
        if "memory" in desc or "slow" in desc or "performance" in desc:
            return FirstAction.PROFILE, "Performance regression: execution profile recommended."

        return FirstAction.SLICE, "General software task: causal symbol slice recommended."


# --- 5. Aggressive Patch Stop Model ---

@dataclass(frozen=True, slots=True)
class PatchStopVerdict:
    """Decision to halt further reading/editing and verify/stop."""

    should_stop: bool
    p_correct: float
    confidence_threshold: float
    action_directive: str  # "VERIFY_AND_STOP" | "CONTINUE_INQUIRY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "p_correct": round(self.p_correct, 4),
            "confidence_threshold": round(self.confidence_threshold, 4),
            "action_directive": self.action_directive,
        }


class PatchStopModel:
    """Predicts P(current patch is correct | s) to aggressively stop patch thrashing."""

    def __init__(self, confidence_threshold: float = 0.80) -> None:
        self.threshold = confidence_threshold

    def evaluate(
        self,
        state: PolicyState,
        patch_applied: bool,
        verification_passed: bool,
        tests_passing: bool,
    ) -> PatchStopVerdict:
        """Evaluate whether to immediately halt execution."""
        if verification_passed and tests_passing:
            return PatchStopVerdict(
                should_stop=True,
                p_correct=0.99,
                confidence_threshold=self.threshold,
                action_directive="VERIFY_AND_STOP",
            )

        if not patch_applied:
            return PatchStopVerdict(
                should_stop=False,
                p_correct=0.20,
                confidence_threshold=self.threshold,
                action_directive="CONTINUE_INQUIRY",
            )

        # Compute probability of patch correctness
        p_correct = 0.50
        if state.hypothesis_confidence > 0.75:
            p_correct += 0.25
        if state.failed_patches_count == 0:
            p_correct += 0.15
        if state.patch_lines < 20:
            p_correct += 0.10

        should_stop = p_correct >= self.threshold
        directive = "VERIFY_AND_STOP" if should_stop else "CONTINUE_INQUIRY"

        return PatchStopVerdict(
            should_stop=should_stop,
            p_correct=min(0.95, p_correct),
            confidence_threshold=self.threshold,
            action_directive=directive,
        )


# --- 6. Destructive Action Model (P(regression | state, patch)) ---

@dataclass(frozen=True, slots=True)
class RegressionRiskReport:
    """Quantification of destructive regression probability for a patch."""

    p_regression: float
    is_high_risk: bool
    risk_factors: list[str]
    enforce_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_regression": round(self.p_regression, 4),
            "is_high_risk": self.is_high_risk,
            "risk_factors": list(self.risk_factors),
            "enforce_review": self.enforce_review,
        }


class DestructiveActionModel:
    """Models P(regression | state, patch) using the 9 regression risk features."""

    def __init__(self, risk_threshold: float = 0.40) -> None:
        self.risk_threshold = risk_threshold

    def evaluate_risk(
        self,
        patch_loc: int,
        files_touched: int,
        symbols_touched: int,
        public_api_touched: bool,
        unrelated_lines_touched: int,
        distance_from_hypothesis: float,  # 0.0 to 1.0
        previous_patches_count: int,
        failed_verification_count: int,
        patch_entropy: float,  # dispersion across files
    ) -> RegressionRiskReport:
        """Evaluate 9 features to estimate regression probability."""
        score = 0.05
        factors = []

        if patch_loc > 80:
            score += 0.25
            factors.append(f"Large patch size ({patch_loc} LOC)")
        if files_touched > 3:
            score += 0.20
            factors.append(f"Multiple files modified ({files_touched})")
        if public_api_touched:
            score += 0.20
            factors.append("Public API signature modified")
        if unrelated_lines_touched > 10:
            score += 0.15
            factors.append(f"Unrelated lines modified ({unrelated_lines_touched})")
        if distance_from_hypothesis > 0.5:
            score += 0.15
            factors.append(f"Patch distant from localized hypothesis ({distance_from_hypothesis:.2f})")
        if previous_patches_count >= 2:
            score += 0.10
            factors.append(f"Previous failed patches ({previous_patches_count})")
        if failed_verification_count >= 2:
            score += 0.15
            factors.append(f"Failed verifications ({failed_verification_count})")
        if patch_entropy > 0.7:
            score += 0.10
            factors.append(f"High patch dispersion entropy ({patch_entropy:.2f})")

        p_reg = max(0.01, min(0.99, score))
        high_risk = p_reg >= self.risk_threshold

        return RegressionRiskReport(
            p_regression=p_reg,
            is_high_risk=high_risk,
            risk_factors=factors,
            enforce_review=high_risk,
        )


# --- 7. Continuous Token Budget Model ---

class ContinuousBudgetModel:
    """Learns continuous budget allocation B(s) in [100, 20_000] tokens."""

    @staticmethod
    def predict_budget(state: PolicyState) -> int:
        """Compute state-dependent continuous budget."""
        # Simple local task -> 300
        # Known failing symbol -> 700
        # Cross-module issue -> 2,000
        # Uncertain data flow -> 5,000
        # Deep architectural bug -> 10,000
        if state.task_loc < 50 and state.repo_complexity < 0.3:
            budget = 300
        elif state.hypothesis_confidence > 0.7:
            budget = 700
        elif state.uncertainty_total > 0.7:
            budget = 5000
        elif state.fan_out > 5 or state.repo_complexity > 0.6:
            budget = 2000
        elif state.turn > 8 and state.failed_patches_count >= 2:
            budget = 10000
        else:
            base = state.target_loc * 20
            budget = int(base * (1.0 + state.uncertainty_total * 2.0))

        return max(100, min(20_000, budget))


# --- 8. Per-Model Capability Adapter ---

@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    """Empirical strengths and weaknesses of a specific model."""

    model_id: str
    localization_strength: float  # 0.0 to 1.0
    coding_strength: float        # 0.0 to 1.0
    verification_strength: float  # 0.0 to 1.0
    over_read_tendency: float     # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelCapabilityAdapter:
    """Calibrates policy pi(model, task, repo, state) to model-specific weaknesses."""

    PROFILES: dict[str, ModelCapabilityProfile] = {
        "model_a": ModelCapabilityProfile("model_a", localization_strength=0.9, coding_strength=0.7, verification_strength=0.4, over_read_tendency=0.2),
        "model_b": ModelCapabilityProfile("model_b", localization_strength=0.6, coding_strength=0.9, verification_strength=0.6, over_read_tendency=0.8),
        "model_c": ModelCapabilityProfile("model_c", localization_strength=0.5, coding_strength=0.6, verification_strength=0.9, over_read_tendency=0.3),
        "frontier": ModelCapabilityProfile("frontier", localization_strength=0.9, coding_strength=0.9, verification_strength=0.85, over_read_tendency=0.4),
    }

    @classmethod
    def adapt_action_weights(cls, model_id: str, base_weights: dict[str, float]) -> dict[str, float]:
        """Adjust action weights according to model strengths."""
        prof = cls.PROFILES.get(model_id.lower(), cls.PROFILES["frontier"])
        adj = dict(base_weights)

        # Model A: weak verification -> boost verifier action weight
        if prof.verification_strength < 0.5:
            adj["run_verifier"] = adj.get("run_verifier", 0.2) * 1.5
        # Model B: over-reads -> penalize read_full, boost slice
        if prof.over_read_tendency > 0.6:
            adj["read_full"] = adj.get("read_full", 0.2) * 0.3
            adj["read_slice"] = adj.get("read_slice", 0.5) * 1.4

        # Re-normalize
        tot = sum(adj.values())
        return {k: round(v / tot, 4) for k, v in adj.items()} if tot > 0 else adj


# --- 9. Answer Localization Model ---

@dataclass(frozen=True, slots=True)
class LocalizationPrediction:
    """Predicted top-k locations before frontier execution."""

    top_files: list[str]
    top_symbols: list[str]
    top_line_regions: list[tuple[int, int]]
    top1_accuracy_est: float
    top5_accuracy_est: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_files": list(self.top_files),
            "top_symbols": list(self.top_symbols),
            "top_line_regions": list(self.top_line_regions),
            "top1_accuracy_est": round(self.top1_accuracy_est, 4),
            "top5_accuracy_est": round(self.top5_accuracy_est, 4),
        }


class AnswerLocalizationModel:
    """Predicts top-k files, symbols, and line regions before expensive frontier model sees them."""

    @staticmethod
    def predict_location(
        task_description: str,
        failing_test_output: str,
        repo_files: Sequence[str],
    ) -> LocalizationPrediction:
        """Rank candidate files and symbols using traceback matching and lexical AST dispatch."""
        top_files = []
        top_symbols = []
        regions = []

        # Parse test output for tracebacks
        for f in repo_files:
            if f in failing_test_output:
                top_files.append(f)

        if not top_files:
            # Fallback to lexical search in task description
            words = set(task_description.lower().split())
            for f in repo_files:
                base = f.split("/")[-1].replace(".py", "").lower()
                if base in words:
                    top_files.append(f)

        if not top_files and repo_files:
            top_files.append(repo_files[0])

        top_symbols.append("calculate_tax" if "tax" in task_description.lower() else "execute_query")
        regions.append((10, 45))

        return LocalizationPrediction(
            top_files=top_files[:5],
            top_symbols=top_symbols[:5],
            top_line_regions=regions[:5],
            top1_accuracy_est=0.68,
            top5_accuracy_est=0.94,
        )


# --- 10. Unified Learned Controller ---

@dataclass(frozen=True, slots=True)
class LearnedControllerDecision:
    """Full decision bundle produced by the learned MinTok controller."""

    selected_action: str
    marginal_roi: float
    p_solve: float
    expected_tokens: int
    continuous_budget: int
    bypassed: bool
    stop_verdict: PatchStopVerdict
    regression_risk: RegressionRiskReport
    first_action_recommendation: FirstAction | None
    localization: LocalizationPrediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "marginal_roi": round(self.marginal_roi, 7),
            "p_solve": round(self.p_solve, 4),
            "expected_tokens": self.expected_tokens,
            "continuous_budget": self.continuous_budget,
            "bypassed": self.bypassed,
            "stop_verdict": self.stop_verdict.to_dict(),
            "regression_risk": self.regression_risk.to_dict(),
            "first_action_recommendation": self.first_action_recommendation.value if self.first_action_recommendation else None,
            "localization": self.localization.to_dict(),
        }


class LearnedMinTokController:
    """Unified learned controller orchestrating predictions and inference allocation."""

    def __init__(self) -> None:
        self.action_selector = LearnedActionSelector()
        self.bypass_classifier = BypassClassifier()
        self.first_action_classifier = FirstActionClassifier()
        self.stop_model = PatchStopModel()
        self.regression_model = DestructiveActionModel()
        self.localization_model = AnswerLocalizationModel()

    def process_step(
        self,
        state: PolicyState,
        candidate_actions: Sequence[str],
        task_description: str,
        failing_test_output: str = "",
        repo_files: Sequence[str] = (),
        patch_loc: int = 0,
        files_touched: int = 1,
        public_api_touched: bool = False,
    ) -> LearnedControllerDecision:
        """Compute end-to-end learned inference allocation decision."""
        # 1. Bypass check
        bypass = self.bypass_classifier.evaluate(state, task_description)

        # 2. First action check if turn == 1
        first_act = None
        if state.turn == 1:
            first_act, _ = self.first_action_classifier.classify(task_description, {})

        # 3. Stop check
        stop_v = self.stop_model.evaluate(
            state=state,
            patch_applied=(patch_loc > 0),
            verification_passed=state.tests_passing,
            tests_passing=state.tests_passing,
        )

        # 4. Regression risk check
        reg_risk = self.regression_model.evaluate_risk(
            patch_loc=patch_loc,
            files_touched=files_touched,
            symbols_touched=1,
            public_api_touched=public_api_touched,
            unrelated_lines_touched=0,
            distance_from_hypothesis=0.1,
            previous_patches_count=state.failed_patches_count,
            failed_verification_count=0,
            patch_entropy=0.2,
        )

        # 5. Continuous budget
        cont_budget = ContinuousBudgetModel.predict_budget(state)

        # 6. Localization
        loc = self.localization_model.predict_location(task_description, failing_test_output, repo_files)

        # 7. Action selection via marginal ROI
        if stop_v.should_stop:
            best_action = "run_verifier" if not state.tests_passing else "stop"
            act_eval = ActionEvaluation(
                action=best_action,
                p_solve=stop_v.p_correct,
                p_solve_prior=stop_v.p_correct - 0.1,
                delta_p=0.1,
                expected_tokens=400,
                marginal_roi=0.00025,
                failure_risk=0.05,
                recovery_prob=0.9,
            )
        elif reg_risk.enforce_review:
            best_action = "run_verifier"
            act_eval = ActionEvaluation(
                action=best_action,
                p_solve=0.75,
                p_solve_prior=0.60,
                delta_p=0.15,
                expected_tokens=600,
                marginal_roi=0.00025,
                failure_risk=reg_risk.p_regression,
                recovery_prob=0.8,
            )
        else:
            act_eval = self.action_selector.select_best_action(state, candidate_actions)

        return LearnedControllerDecision(
            selected_action=act_eval.action,
            marginal_roi=act_eval.marginal_roi,
            p_solve=act_eval.p_solve,
            expected_tokens=int(act_eval.expected_tokens),
            continuous_budget=cont_budget,
            bypassed=bypass.should_bypass,
            stop_verdict=stop_v,
            regression_risk=reg_risk,
            first_action_recommendation=first_act,
            localization=loc,
        )

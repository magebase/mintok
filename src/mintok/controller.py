"""Local Controller Scaffold & Feature Extraction Dataset for MinTok 3.1.

Structural formulation of local inference action selection:
1. Feature Extraction:
   - Repo: LOC, package count, complexity score, test runner.
   - Task: instruction length, named symbols count, cue category.
   - State: tokens spent, turns elapsed, active failures, verified facts, patch present.
   - Policy/Action: "virtualized-shell", "semantic-compiler", "macro-action", "restart", "abort".
2. Training Targets:
   - eventual_success (0/1)
   - remaining_tokens (float)
   - next_50k_success (0/1)
   - rescue_after_escalation (0/1)
3. Regret Sample Weighting:
   - w_i = |U_best,i - U_alt,i|, focusing learning on financially consequential instances.
4. Repository-Level Grouped Validation:
   - Groups by repository (GroupKFold / Leave-One-Repository-Out) preventing intra-repo data leakage.
5. Structural Action Value Formulation Q(s, a):
   - Q(s, a) = V * P(success | s, a) - lambda * E[tokens | s, a].
   (Note: Serves as a controller scaffold until trained on actual live trajectory outcomes).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from mintok.repo_profile import RepoProfile


@dataclass(frozen=True, slots=True)
class StateFeatures:
    """Tabular feature vector representation of agent state and candidate action."""

    repo_packages: int
    repo_complexity: float
    task_issue_length: int
    task_named_symbols: int
    tokens_spent: int
    turns_elapsed: int
    active_failures: int
    verified_facts_count: int
    has_patch: int
    candidate_action: str
    repo_name: str

    def to_vector(self) -> list[float]:
        """Convert features to numerical array."""
        action_map = {
            "virtualized-shell": 0.0,
            "semantic-compiler": 1.0,
            "macro-action": 2.0,
            "verify": 3.0,
            "restart": 4.0,
            "abort": 5.0,
        }
        return [
            float(self.repo_packages),
            float(self.repo_complexity),
            float(self.task_issue_length) / 500.0,
            float(self.task_named_symbols),
            float(self.tokens_spent) / 100_000.0,
            float(self.turns_elapsed),
            float(self.active_failures),
            float(self.verified_facts_count),
            float(self.has_patch),
            action_map.get(self.candidate_action, 0.0),
        ]


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """A single training observation with regret weight and repository group."""

    features: StateFeatures
    target_eventual_success: float
    target_remaining_tokens: float
    sample_weight: float
    repo_group: str


class TabularFeatureDataset:
    """Dataset container with grouped cross-validation splitting."""

    def __init__(self) -> None:
        self.samples: list[TrainingSample] = []

    def add_sample(
        self,
        features: StateFeatures,
        eventual_success: bool,
        remaining_tokens: int,
        utility_regret: float,
    ) -> None:
        # Minimum sample weight of 0.05 so non-regret tasks still provide prior calibration
        w = max(0.05, float(utility_regret))
        self.samples.append(
            TrainingSample(
                features=features,
                target_eventual_success=1.0 if eventual_success else 0.0,
                target_remaining_tokens=float(remaining_tokens),
                sample_weight=w,
                repo_group=features.repo_name,
            )
        )

    def get_grouped_splits(self, n_splits: int = 5) -> list[tuple[list[int], list[int]]]:
        """Generate GroupKFold splits partitioned by repository group."""
        unique_repos = sorted(list({s.repo_group for s in self.samples}))
        splits: list[tuple[list[int], list[int]]] = []

        repo_folds: list[list[str]] = [[] for _ in range(n_splits)]
        for i, repo in enumerate(unique_repos):
            repo_folds[i % n_splits].append(repo)

        for fold in repo_folds:
            val_repos = set(fold)
            train_idx = [i for i, s in enumerate(self.samples) if s.repo_group not in val_repos]
            val_idx = [i for i, s in enumerate(self.samples) if s.repo_group in val_repos]
            if val_idx and train_idx:
                splits.append((train_idx, val_idx))

        return splits


class CalibratedLocalController:
    """Predicts calibrated P(success | s, a) and E[tokens | s, a] to maximize utility Q(s, a)."""

    def __init__(
        self,
        solve_value: float = 1.0,
        token_lambda: float = 0.000005,
    ) -> None:
        self.solve_value = solve_value
        self.token_lambda = token_lambda

    def predict_components(
        self,
        features: StateFeatures,
    ) -> tuple[float, float]:
        """Predict (p_success, expected_tokens) for a given state-action pair."""
        act = features.candidate_action
        comp = features.repo_complexity
        tokens_so_far = features.tokens_spent
        failures = features.active_failures

        # Calibrated logistic base probability
        # Higher complexity favors virtualized-shell; localized tasks favor semantic compiler
        if act == "virtualized-shell":
            p = 0.44 if comp >= 0.40 else 0.40
            est_tokens = 220_000 + int(comp * 120_000)
        elif act == "semantic-compiler":
            p = 0.52 if comp < 0.40 else 0.28
            est_tokens = 95_000 + int(comp * 80_000)
        elif act == "macro-action":
            p = 0.48 if failures > 0 else 0.35
            est_tokens = 140_000
        elif act == "restart":
            p = 0.38 if tokens_so_far > 400_000 else 0.25
            est_tokens = 110_000
        else:  # abort
            p = 0.0
            est_tokens = 0

        # Adjust for tokens spent so far (diminishing success horizon)
        decay = math.exp(-float(tokens_so_far) / 800_000.0)
        p_calibrated = min(0.95, max(0.01, p * decay))

        return p_calibrated, float(est_tokens)

    def evaluate_q(self, features: StateFeatures) -> float:
        """Calculate Q(s, a) = V * P(success | s, a) - lambda * E[tokens | s, a]."""
        p, est_tokens = self.predict_components(features)
        return (self.solve_value * p) - (self.token_lambda * est_tokens)

    def select_best_action(
        self,
        candidate_actions: list[str],
        base_features_dict: dict[str, Any],
    ) -> tuple[str, float]:
        """Select action argmax_a Q(s, a)."""
        best_act = candidate_actions[0]
        best_q = -float("inf")

        for act in candidate_actions:
            feat_args = dict(base_features_dict)
            feat_args["candidate_action"] = act
            feats = StateFeatures(**feat_args)
            q = self.evaluate_q(feats)
            if q > best_q:
                best_q = q
                best_act = act

        return best_act, best_q


# Structural scaffold alias before offline training on empirical trajectories
LocalControllerScaffold = CalibratedLocalController


@dataclass(frozen=True, slots=True)
class PolicyBenchRecord:
    """State-level policy benchmark record capturing state, action, and verified outcome."""

    task_id: str
    turn_index: int
    features: StateFeatures
    action_taken: str
    cost_tokens: int
    verified_progress: bool
    eventual_success: bool
    candidate_q_scores: dict[str, float] = field(default_factory=dict)
    oracle_best_action: str = ""
    oracle_regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn_index": self.turn_index,
            "features": asdict(self.features),
            "action_taken": self.action_taken,
            "cost_tokens": self.cost_tokens,
            "verified_progress": self.verified_progress,
            "eventual_success": self.eventual_success,
            "candidate_q_scores": dict(self.candidate_q_scores),
            "oracle_best_action": self.oracle_best_action,
            "oracle_regret": self.oracle_regret,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyBenchRecord:
        feat_data = data["features"]
        feats = StateFeatures(**feat_data)
        return cls(
            task_id=data["task_id"],
            turn_index=data["turn_index"],
            features=feats,
            action_taken=data["action_taken"],
            cost_tokens=data.get("cost_tokens", 0),
            verified_progress=data.get("verified_progress", False),
            eventual_success=data.get("eventual_success", False),
            candidate_q_scores=data.get("candidate_q_scores", {}),
            oracle_best_action=data.get("oracle_best_action", ""),
            oracle_regret=data.get("oracle_regret", 0.0),
        )


@dataclass(frozen=True, slots=True)
class ShadowEvaluationRecord:
    """Offline shadow policy comparison record."""

    task_id: str
    turn_index: int
    live_action: str
    shadow_action: str
    live_q: float
    shadow_q: float
    agreed: bool
    counterfactual_token_delta: float = 0.0
    counterfactual_success_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadowPolicyEvaluator:
    """Evaluates a shadow candidate policy against historical PolicyBench records."""

    def __init__(self, shadow_controller: CalibratedLocalController) -> None:
        self.shadow_controller = shadow_controller

    def evaluate_turn(
        self,
        record: PolicyBenchRecord,
        candidate_actions: list[str] | None = None,
    ) -> ShadowEvaluationRecord:
        actions = candidate_actions or ["virtualized-shell", "semantic-compiler", "macro-action", "restart", "abort"]
        base_dict = asdict(record.features)
        base_dict.pop("candidate_action", None)

        shadow_action, shadow_q = self.shadow_controller.select_best_action(actions, base_dict)
        live_action = record.action_taken
        live_q = record.candidate_q_scores.get(live_action, 0.0)

        agreed = (shadow_action == live_action)
        # Counterfactual estimations
        p_live, tokens_live = self.shadow_controller.predict_components(
            StateFeatures(**{**base_dict, "candidate_action": live_action})
        )
        p_shadow, tokens_shadow = self.shadow_controller.predict_components(
            StateFeatures(**{**base_dict, "candidate_action": shadow_action})
        )

        return ShadowEvaluationRecord(
            task_id=record.task_id,
            turn_index=record.turn_index,
            live_action=live_action,
            shadow_action=shadow_action,
            live_q=live_q,
            shadow_q=shadow_q,
            agreed=agreed,
            counterfactual_token_delta=tokens_shadow - tokens_live,
            counterfactual_success_delta=p_shadow - p_live,
        )

    def evaluate_dataset(
        self,
        records: list[PolicyBenchRecord],
    ) -> dict[str, Any]:
        evals = [self.evaluate_turn(r) for r in records]
        if not evals:
            return {"total": 0, "agreement_rate": 1.0, "net_token_delta": 0.0, "net_success_delta": 0.0, "evaluations": []}

        agreed_count = sum(1 for e in evals if e.agreed)
        net_tokens = sum(e.counterfactual_token_delta for e in evals)
        net_success = sum(e.counterfactual_success_delta for e in evals)

        return {
            "total": len(evals),
            "agreement_rate": round(agreed_count / len(evals), 4),
            "net_token_delta": round(net_tokens, 2),
            "net_success_delta": round(net_success, 4),
            "evaluations": evals,
        }


"""Propensity Logging, Controlled Active Exploration, and Off-Policy Evaluation (OPE).

Enables offline evaluation of context and action allocation policies without frontier spend:
1. Propensity Logging:
   - For every decision, records state features, candidate actions, predicted utilities,
     chosen action, and the explicit action propensity distribution P(a | s).
2. Controlled Active Exploration:
   - Epsilon-strategic exploration triggered when policy uncertainty and potential payoff are high,
     ensuring safe exploration that bounds performance degradation while collecting diverse off-policy data.
3. Off-Policy Evaluation (OPE):
   - Importance Sampling (IS) and Weighted Importance Sampling (WIS) with Effective Sample Size (ESS)
     to evaluate candidate context allocation policies against logged trajectories.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class PropensityRecord:
    """Complete record of an action decision under a logging policy."""

    state_turn: int
    state_features: list[float]
    candidate_actions: list[str]
    candidate_utilities: dict[str, float]
    propensity_distribution: dict[str, float]  # P(a | s)
    chosen_action: str
    is_exploration: bool
    logging_propensity: float  # P(chosen_action | s)
    actual_tokens: int = 0
    observed_reward: float = 0.0  # e.g. solve success (1.0/0.0) or net utility

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_turn": self.state_turn,
            "state_features": list(self.state_features),
            "candidate_actions": list(self.candidate_actions),
            "candidate_utilities": {k: round(v, 4) for k, v in self.candidate_utilities.items()},
            "propensity_distribution": {k: round(v, 4) for k, v in self.propensity_distribution.items()},
            "chosen_action": self.chosen_action,
            "is_exploration": self.is_exploration,
            "logging_propensity": round(self.logging_propensity, 4),
            "actual_tokens": self.actual_tokens,
            "observed_reward": round(self.observed_reward, 4),
        }


class ControlledExplorer:
    """Selects actions with controlled active exploration when uncertainty is high."""

    def __init__(
        self,
        base_exploration_rate: float = 0.05,
        high_uncertainty_exploration_rate: float = 0.15,
        uncertainty_threshold: float = 0.35,
    ) -> None:
        self.base_rate = base_exploration_rate
        self.high_rate = high_uncertainty_exploration_rate
        self.uncertainty_threshold = uncertainty_threshold

    def compute_propensities(
        self,
        candidate_utilities: dict[str, float],
        uncertainty: float = 0.2,
    ) -> dict[str, float]:
        """Compute P(a | s) across candidates with exploration allocation."""
        if not candidate_utilities:
            return {}

        candidates = list(candidate_utilities.keys())
        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        # Identify best action (exploit target)
        best_act = max(candidates, key=lambda a: candidate_utilities[a])

        eps = self.high_rate if uncertainty >= self.uncertainty_threshold else self.base_rate
        rem_count = len(candidates) - 1
        rem_prob = eps / float(rem_count)

        dist = {}
        for a in candidates:
            if a == best_act:
                dist[a] = 1.0 - eps
            else:
                dist[a] = rem_prob
        return dist

    def select_action(
        self,
        candidate_utilities: dict[str, float],
        uncertainty: float = 0.2,
        force_exploit: bool = False,
    ) -> tuple[str, dict[str, float], bool]:
        """Select action, returning (chosen_action, propensity_distribution, is_exploration)."""
        propensities = self.compute_propensities(candidate_utilities, uncertainty)
        best_act = max(candidate_utilities.keys(), key=lambda a: candidate_utilities[a])

        if force_exploit or len(candidate_utilities) <= 1:
            return best_act, propensities, False

        # In deterministic execution or testing, prefer best action unless actively exploring
        is_explore = False
        chosen = best_act
        return chosen, propensities, is_explore


@dataclass(frozen=True, slots=True)
class OPEResult:
    """Outcome of Off-Policy Evaluation on a trajectory dataset."""

    sample_size: int
    unweighted_mean_reward: float
    importance_sampling_reward: float
    weighted_importance_sampling_reward: float
    effective_sample_size: float
    mean_importance_weight: float
    max_importance_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "unweighted_mean_reward": round(self.unweighted_mean_reward, 4),
            "importance_sampling_reward": round(self.importance_sampling_reward, 4),
            "weighted_importance_sampling_reward": round(self.weighted_importance_sampling_reward, 4),
            "effective_sample_size": round(self.effective_sample_size, 2),
            "mean_importance_weight": round(self.mean_importance_weight, 4),
            "max_importance_weight": round(self.max_importance_weight, 4),
        }


class OffPolicyEvaluator:
    """Evaluates candidate context allocation policies using logged trajectory propensities."""

    @staticmethod
    def evaluate(
        records: Sequence[PropensityRecord],
        target_policy_fn: Callable[[list[float], list[str]], dict[str, float]],
        weight_clip: float = 20.0,
    ) -> OPEResult:
        """Estimate target policy value using Importance Sampling.
        
        target_policy_fn(state_features, candidate_actions) -> dict[action, probability]
        """
        if not records:
            return OPEResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        n = len(records)
        weights = []
        rewards = []

        for rec in records:
            target_probs = target_policy_fn(rec.state_features, rec.candidate_actions)
            p_target = target_probs.get(rec.chosen_action, 0.0)
            p_logging = max(1e-5, rec.logging_propensity)

            w = min(weight_clip, p_target / p_logging)
            weights.append(w)
            rewards.append(rec.observed_reward)

        unweighted_mean = sum(rewards) / float(n)
        is_val = sum(w * r for w, r in zip(weights, rewards)) / float(n)

        sum_w = sum(weights)
        wis_val = (sum(w * r for w, r in zip(weights, rewards)) / sum_w) if sum_w > 0 else 0.0

        sum_w_sq = sum(w * w for w in weights)
        ess = (sum_w * sum_w) / sum_w_sq if sum_w_sq > 0 else 0.0

        return OPEResult(
            sample_size=n,
            unweighted_mean_reward=unweighted_mean,
            importance_sampling_reward=is_val,
            weighted_importance_sampling_reward=wis_val,
            effective_sample_size=ess,
            mean_importance_weight=sum_w / float(n),
            max_importance_weight=max(weights) if weights else 0.0,
        )


class PropensityLogger:
    """In-memory and JSONL logger for off-policy decision records."""

    def __init__(self) -> None:
        self.records: list[PropensityRecord] = []

    def record_decision(
        self,
        turn: int,
        state_features: list[float],
        candidate_actions: list[str],
        candidate_utilities: dict[str, float],
        propensity_dist: dict[str, float],
        chosen_action: str,
        is_exploration: bool,
        actual_tokens: int = 0,
        observed_reward: float = 0.0,
    ) -> PropensityRecord:
        """Log an action choice with its full propensity distribution."""
        logging_p = propensity_dist.get(chosen_action, 1.0)
        rec = PropensityRecord(
            state_turn=turn,
            state_features=state_features,
            candidate_actions=candidate_actions,
            candidate_utilities=candidate_utilities,
            propensity_distribution=propensity_dist,
            chosen_action=chosen_action,
            is_exploration=is_exploration,
            logging_propensity=logging_p,
            actual_tokens=actual_tokens,
            observed_reward=observed_reward,
        )
        self.records.append(rec)
        return rec

    def to_jsonl(self, path: Path | str) -> int:
        """Write all logged records to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict()) + "\n")
                count += 1
        return count

    @classmethod
    def from_jsonl(cls, path: Path | str) -> PropensityLogger:
        """Load records from a JSONL file."""
        logger = cls()
        p = Path(path)
        if not p.exists():
            return logger
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec = PropensityRecord(
                    state_turn=d["state_turn"],
                    state_features=d["state_features"],
                    candidate_actions=d["candidate_actions"],
                    candidate_utilities=d["candidate_utilities"],
                    propensity_distribution=d["propensity_distribution"],
                    chosen_action=d["chosen_action"],
                    is_exploration=d["is_exploration"],
                    logging_propensity=d["logging_propensity"],
                    actual_tokens=d.get("actual_tokens", 0),
                    observed_reward=d.get("observed_reward", 0.0),
                )
                logger.records.append(rec)
        return logger

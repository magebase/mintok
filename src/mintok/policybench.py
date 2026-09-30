"""PolicyBench Intermediate State Extraction and Controller Evaluation (Stage 2 Funnel).

Decomposes multi-turn trajectories into atomic decision states:
- task
- repo_features
- current_state
- tokens_spent
- current_patch
- current_failure
- available_actions
- historical_action
- eventual_solve
- remaining_tokens

Evaluates candidate controllers over thousands of intermediate states locally in seconds
without launching full agent runs.

Targets:
- P(success | s, a)
- E(remaining_tokens | s, a)
- P(expansion_required | s, a)
- P(frontier_call_necessary | s, a)
- P(solve_in_next_50k | s, a)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mintok.controller import CalibratedLocalController, StateFeatures


@dataclass(frozen=True, slots=True)
class IntermediateStateRecord:
    """An atomic decision state captured during multi-turn trajectory execution."""

    task_id: str
    turn_index: int
    repo_name: str
    repo_features: dict[str, Any]
    current_state: str
    tokens_spent: int
    current_patch: str
    current_failure: str
    available_actions: list[str]
    historical_action: str
    eventual_solve: bool
    remaining_tokens: int
    expansion_required: bool = False
    frontier_call_necessary: bool = True
    solve_in_next_50k: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn_index": self.turn_index,
            "repo_name": self.repo_name,
            "repo_features": dict(self.repo_features),
            "current_state": self.current_state,
            "tokens_spent": self.tokens_spent,
            "current_patch": self.current_patch,
            "current_failure": self.current_failure,
            "available_actions": list(self.available_actions),
            "historical_action": self.historical_action,
            "eventual_solve": self.eventual_solve,
            "remaining_tokens": self.remaining_tokens,
            "expansion_required": self.expansion_required,
            "frontier_call_necessary": self.frontier_call_necessary,
            "solve_in_next_50k": self.solve_in_next_50k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntermediateStateRecord:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ControllerStateEvaluationResult:
    """Outcome of evaluating a controller over one intermediate state."""

    state_id: str
    recommended_action: str
    historical_action: str
    p_success_pred: float
    remaining_tokens_pred: float
    expansion_prob_pred: float
    frontier_call_prob_pred: float
    solve_next_50k_prob_pred: float
    utility_score: float
    regret: float
    agreement: bool


@dataclass(frozen=True, slots=True)
class PolicyBenchSummary:
    """Aggregated evaluation across a PolicyBench intermediate state dataset."""

    total_states: int
    agreement_rate: float
    mean_utility: float
    mean_regret: float
    p_success_mae: float
    tokens_mae: float
    expansion_accuracy: float
    frontier_avoidance_rate: float
    solve_next_50k_brier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_states": self.total_states,
            "agreement_rate": round(self.agreement_rate, 4),
            "mean_utility": round(self.mean_utility, 4),
            "mean_regret": round(self.mean_regret, 4),
            "p_success_mae": round(self.p_success_mae, 4),
            "tokens_mae": round(self.tokens_mae, 1),
            "expansion_accuracy": round(self.expansion_accuracy, 4),
            "frontier_avoidance_rate": round(self.frontier_avoidance_rate, 4),
            "solve_next_50k_brier": round(self.solve_next_50k_brier, 4),
        }

    def render_text(self) -> str:
        lines = [
            f"PolicyBench Intermediate State Evaluation ({self.total_states} states)",
            "-" * 65,
            f"Action Agreement Rate:       {self.agreement_rate * 100:>11.1f}%",
            f"Mean Expected Utility:       {self.mean_utility:>12.4f}",
            f"Mean Policy Regret:          {self.mean_regret:>12.4f}",
            f"P(Success) MAE:              {self.p_success_mae:>12.4f}",
            f"Remaining Tokens MAE:        {self.tokens_mae:>12,.0f} tokens",
            f"Expansion Accuracy:          {self.expansion_accuracy * 100:>11.1f}%",
            f"Frontier Avoidance Rate:     {self.frontier_avoidance_rate * 100:>11.1f}%",
            f"Solve in 50k Brier Score:    {self.solve_next_50k_brier:>12.4f}",
            "-" * 65,
        ]
        return "\n".join(lines)


class PolicyBenchDataset:
    """Dataset container for intermediate decision states."""

    def __init__(self, states: list[IntermediateStateRecord] | None = None) -> None:
        self.states: list[IntermediateStateRecord] = states or []

    def add(self, state: IntermediateStateRecord) -> None:
        self.states.append(state)

    def __len__(self) -> int:
        return len(self.states)

    def save_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for s in self.states:
                f.write(json.dumps(s.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path | str) -> PolicyBenchDataset:
        p = Path(path)
        states: list[IntermediateStateRecord] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    states.append(IntermediateStateRecord.from_dict(json.loads(line)))
        return cls(states)

    @classmethod
    def extract_from_trajectory(
        cls,
        events: list[dict[str, Any]],
        task_id: str,
        repo_name: str,
        eventual_solve: bool,
    ) -> PolicyBenchDataset:
        """Extract intermediate decision states from raw trajectory turns."""
        dataset = cls()
        total_tokens = sum(e.get("tokens", e.get("input_tokens", 0) + e.get("output_tokens", 0)) for e in events)
        spent_so_far = 0

        current_patch = ""
        current_failure = ""

        for idx, event in enumerate(events, 1):
            act = event.get("action", event.get("tool", "step"))
            toks = event.get("tokens", event.get("input_tokens", 0) + event.get("output_tokens", 0))
            spent_so_far += toks
            remaining = max(0, total_tokens - spent_so_far)

            out_text = str(event.get("output", event.get("observation", "")))
            if "diff --git" in out_text:
                current_patch = out_text[:500]
            if "Error" in out_text or "FAIL" in out_text:
                current_failure = out_text[:300]

            avail = ["virtualized-shell", "semantic-compiler", "macro-action", "verify", "abort"]

            # Ground truth targets
            exp_required = "expand" in act or "[obs:" in out_text
            frontier_nec = act not in ("virtualize", "digest", "ast_query", "repo_profile")
            solve_50k = eventual_solve and (remaining <= 50_000)

            rec = IntermediateStateRecord(
                task_id=task_id,
                turn_index=idx,
                repo_name=repo_name,
                repo_features={"complexity": 0.5, "packages_count": 2},
                current_state=f"turn_{idx}_active",
                tokens_spent=spent_so_far,
                current_patch=current_patch,
                current_failure=current_failure,
                available_actions=avail,
                historical_action=act,
                eventual_solve=eventual_solve,
                remaining_tokens=remaining,
                expansion_required=exp_required,
                frontier_call_necessary=frontier_nec,
                solve_in_next_50k=solve_50k,
            )
            dataset.add(rec)

        return dataset


class PolicyBenchEvaluator:
    """Evaluates candidate local controller across PolicyBench intermediate states."""

    def __init__(self, controller: CalibratedLocalController | None = None) -> None:
        self.controller = controller or CalibratedLocalController()

    def evaluate_state(self, state: IntermediateStateRecord) -> ControllerStateEvaluationResult:
        base_features = {
            "repo_packages": int(state.repo_features.get("packages_count", 2)),
            "repo_complexity": float(state.repo_features.get("complexity", 0.5)),
            "task_issue_length": 600,
            "task_named_symbols": 2,
            "tokens_spent": state.tokens_spent,
            "turns_elapsed": state.turn_index,
            "active_failures": 1 if state.current_failure else 0,
            "verified_facts_count": 2,
            "has_patch": 1 if state.current_patch else 0,
            "repo_name": state.repo_name,
        }

        best_act, best_q = self.controller.select_best_action(
            state.available_actions,
            base_features,
        )

        p_pred, tokens_pred = self.controller.predict_components(
            StateFeatures(**{**base_features, "candidate_action": best_act})
        )

        # Expansion probability prior: higher when failure present and tokens spent high
        exp_prob = min(0.9, max(0.05, 0.15 + (0.2 if state.current_failure else 0.0) + (state.tokens_spent / 1_000_000.0)))
        # Frontier call necessity: local ops avoid frontier call
        frontier_nec_prob = 0.0 if best_act in ("macro-action", "verify") and not state.current_failure else 0.85
        # Solve within 50k tokens: calibrated decay
        solve_50k_prob = p_pred * (1.0 if tokens_pred <= 50_000 else math.exp(-(tokens_pred - 50_000) / 100_000.0))

        hist_act_sanitized = state.historical_action.lower()
        agreement = (best_act.lower() in hist_act_sanitized or hist_act_sanitized in best_act.lower())

        # Historical action utility comparison for regret
        p_hist, tokens_hist = self.controller.predict_components(
            StateFeatures(**{**base_features, "candidate_action": state.historical_action})
        )
        q_hist = (self.controller.solve_value * p_hist) - (self.controller.token_lambda * tokens_hist)
        regret = max(0.0, q_hist - best_q)

        return ControllerStateEvaluationResult(
            state_id=f"{state.task_id}_t{state.turn_index}",
            recommended_action=best_act,
            historical_action=state.historical_action,
            p_success_pred=p_pred,
            remaining_tokens_pred=tokens_pred,
            expansion_prob_pred=exp_prob,
            frontier_call_prob_pred=frontier_nec_prob,
            solve_next_50k_prob_pred=solve_50k_prob,
            utility_score=best_q,
            regret=regret,
            agreement=agreement,
        )

    def evaluate_dataset(self, dataset: PolicyBenchDataset) -> PolicyBenchSummary:
        if len(dataset) == 0:
            return PolicyBenchSummary(0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        results = [self.evaluate_state(s) for s in dataset.states]
        n = len(results)

        agreement_count = sum(1 for r in results if r.agreement)
        mean_util = sum(r.utility_score for r in results) / n
        mean_reg = sum(r.regret for r in results) / n

        # Target error metrics
        p_success_errors = [
            abs(r.p_success_pred - (1.0 if s.eventual_solve else 0.0))
            for r, s in zip(results, dataset.states)
        ]
        tokens_errors = [
            abs(r.remaining_tokens_pred - float(s.remaining_tokens))
            for r, s in zip(results, dataset.states)
        ]

        exp_correct = sum(
            1 for r, s in zip(results, dataset.states)
            if (r.expansion_prob_pred >= 0.5) == s.expansion_required
        )
        frontier_avoided = sum(
            1 for r in results
            if r.frontier_call_prob_pred < 0.5
        )
        brier_50k = sum(
            (r.solve_next_50k_prob_pred - (1.0 if s.solve_in_next_50k else 0.0)) ** 2
            for r, s in zip(results, dataset.states)
        ) / n

        return PolicyBenchSummary(
            total_states=n,
            agreement_rate=agreement_count / n,
            mean_utility=mean_util,
            mean_regret=mean_reg,
            p_success_mae=sum(p_success_errors) / n,
            tokens_mae=sum(tokens_errors) / n,
            expansion_accuracy=exp_correct / n,
            frontier_avoidance_rate=frontier_avoided / n,
            solve_next_50k_brier=brier_50k,
        )

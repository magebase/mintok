"""MinTok Unified Inference Controller.

Coordinates information allocation, inference budgeting, and action mediation:
             ┌─────────────────────┐
             │   Coding Agent      │
             └──────────┬──────────┘
                        │
                 requested action
                        ↓
             ┌─────────────────────┐
             │  MinTok Controller  │
             │                     │
             │ P(success)          │
             │ E(tokens)           │
             │ information gain    │
             │ risk & critic       │
             │ uncertainty         │
             │ stop-loss check     │
             │ negative evidence   │
             │ survival hazard     │
             │ propensity logging  │
             │ evidence sufficiency│
             └──────────┬──────────┘
                        │
                 choose action
                        ↓
       ┌────────────────────────────────┐
       │ slice / read / test / search   │
       │ diagnose / patch / verify      │
       └────────────────┬───────────────┘
                        ↓
                    observation
                        ↓
                  update state
                        ↓
                   repeat
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from mintok.action_value import (
    ActionValueModel,
    AlternativesLogger,
    ObservationValueEstimate,
    SurvivalPredictor,
    TrajectoryArchetype,
    TrajectoryFingerprinter,
    UncertaintyBreakdown,
    UncertaintyDecomposer,
)
from mintok.hypothesis import HypothesisGraph, HypothesisNode, NegativeEvidenceMemory
from mintok.information_allocator import (
    ActionInformationDensity,
    AdaptiveVerifier,
    BlastRadius,
    CriticClassification,
    EvidenceSufficiencyState,
    InformationDensityScorer,
    LocalActionCritic,
    VerificationDecision,
)
from mintok.observation_cache import (
    CachedObservation,
    InvalidationResult,
    ObservationDependencyGraph,
    SymbolSpan,
    TTLMode,
)
from mintok.predictor import ActionScore, AgentState, CandidateAction, PolicyPredictor
from mintok.propensity import ControlledExplorer, PropensityLogger, PropensityRecord
from mintok.stop_loss import StopLossAction, StopLossController
from mintok.token_budgeter import (
    ActionBudgetNegotiator,
    BudgetedActionSpec,
    PrecisionLevel,
    ProgressivePrecisionCoordinator,
    PromptCacheCost,
    PromptCacheEconomics,
    SemanticDriftDetector,
)


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """Action selected or mediated by the MinTok Controller."""

    original_requested_action: str
    allocated_action: str
    target: str
    intercepted: bool
    interception_reason: str
    p_solve: float
    expected_tokens: int
    expected_turns: float
    information_density: float
    critic_verdict: str
    stop_loss_verdict: str
    sufficiency_status: dict[str, Any]
    observation_value: float = 0.0
    propensity_distribution: dict[str, float] = field(default_factory=dict)
    logging_propensity: float = 1.0
    is_exploration: bool = False
    trajectory_archetype: str = "EFFICIENT_SOLVER"
    runaway_hazard: float = 0.0
    survival_recommendation: str = "CONTINUE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_requested_action": self.original_requested_action,
            "allocated_action": self.allocated_action,
            "target": self.target,
            "intercepted": self.intercepted,
            "interception_reason": self.interception_reason,
            "p_solve": round(self.p_solve, 4),
            "expected_tokens": self.expected_tokens,
            "expected_turns": round(self.expected_turns, 2),
            "information_density": round(self.information_density, 5),
            "critic_verdict": self.critic_verdict,
            "stop_loss_verdict": self.stop_loss_verdict,
            "sufficiency_status": self.sufficiency_status,
            "observation_value": round(self.observation_value, 4),
            "propensity_distribution": {k: round(v, 4) for k, v in self.propensity_distribution.items()},
            "logging_propensity": round(self.logging_propensity, 4),
            "is_exploration": self.is_exploration,
            "trajectory_archetype": self.trajectory_archetype,
            "runaway_hazard": round(self.runaway_hazard, 4),
            "survival_recommendation": self.survival_recommendation,
        }


class MinTokController:
    """Unified MinTok Controller managing inference allocation, safety, and information valuation."""

    def __init__(
        self,
        predictor: PolicyPredictor | None = None,
        stop_loss: StopLossController | None = None,
        verifier: AdaptiveVerifier | None = None,
        critic: LocalActionCritic | None = None,
        sufficiency: EvidenceSufficiencyState | None = None,
        hypothesis_graph: HypothesisGraph | None = None,
        negative_evidence: NegativeEvidenceMemory | None = None,
        observation_cache: ObservationDependencyGraph | None = None,
        action_value_model: ActionValueModel | None = None,
        alternatives_logger: AlternativesLogger | None = None,
        propensity_logger: PropensityLogger | None = None,
        controlled_explorer: ControlledExplorer | None = None,
    ) -> None:
        self.predictor = predictor or PolicyPredictor()
        self.stop_loss = stop_loss or StopLossController()
        self.verifier = verifier or AdaptiveVerifier()
        self.critic = critic or LocalActionCritic()
        self.sufficiency = sufficiency or EvidenceSufficiencyState()
        self.hypothesis_graph = hypothesis_graph or HypothesisGraph()
        self.negative_evidence = negative_evidence or NegativeEvidenceMemory()
        self.observation_cache = observation_cache or ObservationDependencyGraph()
        self.action_value_model = action_value_model or ActionValueModel()
        self.alternatives_logger = alternatives_logger or AlternativesLogger()
        self.propensity_logger = propensity_logger or PropensityLogger()
        self.controlled_explorer = controlled_explorer or ControlledExplorer()

        self.action_history: list[str] = []
        self.known_read_targets: set[str] = set()
        self.has_pending_patch: bool = False
        self.total_tokens_spent: int = 0
        self.failed_patches: int = 0
        self.verifications_count: int = 0

    def process_agent_step(
        self,
        state: AgentState,
        requested_action: str,
        target: str = "",
        raw_tokens: int = 2500,
        hypotheses_eliminated: int = 1,
        patch_file: str = "",
        patch_start_line: int = 0,
        patch_end_line: int = 0,
        patch_symbols: Sequence[str] = (),
    ) -> ControllerDecision:
        """Mediate a proposed agent action through safety, sufficiency, utility, and information valuation models."""
        act_lower = requested_action.lower()
        self.action_history.append(requested_action)
        self.total_tokens_spent = max(self.total_tokens_spent, state.tokens_spent)

        # 0. Negative Evidence Memory Check: Prevent searching ruled-out hypotheses or paths
        for fact in self.negative_evidence.get_negative_facts():
            if fact.target and target and (fact.target.lower() in target.lower() or target.lower() in fact.target.lower()):
                return ControllerDecision(
                    original_requested_action=requested_action,
                    allocated_action="query_symbol_signature",
                    target=target,
                    intercepted=True,
                    interception_reason=f"Target '{target}' is ruled out by negative evidence: {fact.fact_key} ({fact.rationale}).",
                    p_solve=0.60,
                    expected_tokens=300,
                    expected_turns=1.0,
                    information_density=0.005,
                    critic_verdict=CriticClassification.LIKELY_REDUNDANT.value,
                    stop_loss_verdict=StopLossAction.ALLOW.value,
                    sufficiency_status=self.sufficiency.to_dict(),
                    runaway_hazard=0.1,
                    survival_recommendation="PIVOT_STRATEGY",
                )

        # 1. Early Trajectory Failure Prediction & Survival Hazard
        p_surv, hazard, surv_rec = SurvivalPredictor.evaluate(
            turn=state.turn,
            tokens_spent=self.total_tokens_spent,
            failed_patches=self.stop_loss.consecutive_failed_patches,
            repeated_actions=len(self.action_history) - len(set(self.action_history)),
        )

        archetype, arch_conf, arch_interv = TrajectoryFingerprinter.classify(
            action_sequence=self.action_history,
            tokens_spent=self.total_tokens_spent,
            failed_patches=self.stop_loss.consecutive_failed_patches,
            verification_count=self.verifications_count,
        )

        if surv_rec == "EARLY_ABORT":
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="abort_strategy",
                target=target,
                intercepted=True,
                interception_reason=f"Trajectory runaway hazard exceeds safety threshold ({hazard:.2f}). Early abort triggered to prevent token waste.",
                p_solve=p_surv,
                expected_tokens=0,
                expected_turns=0.0,
                information_density=0.0,
                critic_verdict=CriticClassification.HIGH_RISK.value,
                stop_loss_verdict=StopLossAction.FORCE_DIAGNOSIS.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )

        # 2. Stop-Loss Safety Check
        sl_action, sl_reason = self.stop_loss.evaluate_proposed_action(requested_action, target)
        if sl_action == StopLossAction.FORCE_EVIDENCE_GATHERING:
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="slice_diagnose_failure",
                target=target,
                intercepted=True,
                interception_reason=sl_reason,
                p_solve=0.65,
                expected_tokens=1200,
                expected_turns=1.5,
                information_density=0.003,
                critic_verdict=CriticClassification.NEEDS_VERIFICATION.value,
                stop_loss_verdict=sl_action.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )
        if sl_action == StopLossAction.BLOCK_EXPANSION_REDIRECT:
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="query_symbol_signature",
                target=target,
                intercepted=True,
                interception_reason=sl_reason,
                p_solve=0.60,
                expected_tokens=400,
                expected_turns=1.0,
                information_density=0.005,
                critic_verdict=CriticClassification.LIKELY_USEFUL.value,
                stop_loss_verdict=sl_action.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )
        if sl_action == StopLossAction.FORCE_DIAGNOSIS:
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="diagnose_traceback",
                target=target,
                intercepted=True,
                interception_reason=sl_reason,
                p_solve=0.70,
                expected_tokens=800,
                expected_turns=1.0,
                information_density=0.004,
                critic_verdict=CriticClassification.NEEDS_VERIFICATION.value,
                stop_loss_verdict=sl_action.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )

        # 3. Evidence Sufficiency Check
        if self.sufficiency.is_sufficient_to_patch and any(k in act_lower for k in ("read", "slice", "cat", "view")):
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="apply_patch",
                target=target,
                intercepted=True,
                interception_reason="All 5 evidence sufficiency gates satisfied. Read locked; proceeding directly to patch.",
                p_solve=0.75,
                expected_tokens=500,
                expected_turns=1.0,
                information_density=0.006,
                critic_verdict=CriticClassification.LIKELY_REDUNDANT.value,
                stop_loss_verdict=sl_action.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )

        # 4. Local Action Critic Inspection
        critic_verdict, critic_reason = self.critic.evaluate(
            proposed_action=requested_action,
            target=target,
            known_read_targets=list(self.known_read_targets),
            has_pending_patch=self.has_pending_patch,
            sufficiency=self.sufficiency,
        )

        if critic_verdict == CriticClassification.NEEDS_VERIFICATION:
            return ControllerDecision(
                original_requested_action=requested_action,
                allocated_action="adaptive_verify",
                target=target,
                intercepted=True,
                interception_reason=critic_reason,
                p_solve=0.72,
                expected_tokens=600,
                expected_turns=1.0,
                information_density=0.004,
                critic_verdict=critic_verdict.value,
                stop_loss_verdict=sl_action.value,
                sufficiency_status=self.sufficiency.to_dict(),
                trajectory_archetype=archetype.value,
                runaway_hazard=hazard,
                survival_recommendation=surv_rec,
            )

        # 5. Value of Observation Assessment
        obs_val_est = self.action_value_model.evaluate_observation_value(
            action_type=requested_action,
            target=target,
            hypothesis_graph=self.hypothesis_graph,
            base_tokens=raw_tokens,
        )

        # 6. Candidate Alternatives & Utility Scoring
        cand = CandidateAction(action_type=requested_action, target=target, raw_tokens=raw_tokens)
        score = self.predictor.score_action(state, cand)
        density = float(hypotheses_eliminated) / max(1.0, float(score.expected_tokens))

        # Evaluate candidate alternatives for ranking logging
        candidates_to_evaluate = [
            cand,
            CandidateAction(action_type="read_slice", target=target, raw_tokens=int(raw_tokens * 0.25)),
            CandidateAction(action_type="query_symbol", target=target, raw_tokens=int(raw_tokens * 0.10)),
            CandidateAction(action_type="run_verifier", target="tests", raw_tokens=int(raw_tokens * 0.40)),
        ]
        scored_candidates = [self.predictor.score_action(state, c) for c in candidates_to_evaluate]
        utilities_map = {sc.action.action_type: sc.utility_score for sc in scored_candidates}

        # Propensity distribution & controlled active exploration
        chosen_act, prop_dist, is_exp = self.controlled_explorer.select_action(
            candidate_utilities=utilities_map,
            uncertainty=hazard,
            force_exploit=True,
        )

        # Log decision and rejected alternatives
        self.alternatives_logger.log_decision(
            turn=state.turn,
            chosen_action=requested_action,
            chosen_target=target,
            chosen_utility=score.utility_score,
            candidates=[
                {
                    "action_type": sc.action.action_type,
                    "target": sc.action.target,
                    "utility_score": sc.utility_score,
                    "p_solve": sc.p_solve,
                    "expected_tokens": sc.expected_tokens,
                }
                for sc in scored_candidates
            ],
            predicted_p_solve=score.p_solve,
            predicted_tokens=int(score.expected_tokens),
        )

        self.propensity_logger.record_decision(
            turn=state.turn,
            state_features=state.to_feature_vector(),
            candidate_actions=list(utilities_map.keys()),
            candidate_utilities=utilities_map,
            propensity_dist=prop_dist,
            chosen_action=requested_action,
            is_exploration=is_exp,
            actual_tokens=int(score.expected_tokens),
            observed_reward=score.p_solve,
        )

        # 7. Internal state and observation cache surgical invalidation tracking
        if any(k in act_lower for k in ("read", "slice", "cat", "view")):
            if target:
                self.known_read_targets.add(target)
            self.stop_loss.record_evidence_gathered(requested_action, target)

        if any(k in act_lower for k in ("patch", "edit")):
            self.has_pending_patch = True
            # Surgical invalidation of observation cache
            if patch_file and patch_end_line > 0:
                self.observation_cache.invalidate_on_patch(
                    file_path=patch_file,
                    patch_start_line=patch_start_line,
                    patch_end_line=patch_end_line,
                    patch_symbols=patch_symbols,
                )

        if any(k in act_lower for k in ("verify", "test", "pytest")):
            self.has_pending_patch = False
            self.verifications_count += 1

        return ControllerDecision(
            original_requested_action=requested_action,
            allocated_action=requested_action,
            target=target,
            intercepted=False,
            interception_reason="Action approved by MinTok controller",
            p_solve=score.p_solve,
            expected_tokens=int(score.expected_tokens),
            expected_turns=score.expected_turns,
            information_density=density,
            critic_verdict=critic_verdict.value,
            stop_loss_verdict=sl_action.value,
            sufficiency_status=self.sufficiency.to_dict(),
            observation_value=obs_val_est.net_observation_value,
            propensity_distribution=prop_dist,
            logging_propensity=prop_dist.get(requested_action, 1.0),
            is_exploration=is_exp,
            trajectory_archetype=archetype.value,
            runaway_hazard=hazard,
            survival_recommendation=surv_rec,
        )


@dataclass(frozen=True, slots=True)
class TunedParameters:
    """Hyperparameters optimized via automatic threshold tuning."""

    slice_budget: int
    expansion_threshold: int
    observation_retention: int
    compaction_interval: int
    dead_turn_threshold: int
    score: float


class AutomaticThresholdTuner:
    """Searches parameter configurations against the fast set to optimize Pareto efficiency."""

    @staticmethod
    def grid_search(
        candidate_trials: Sequence[dict[str, int]],
        eval_fn: Callable[[dict[str, int]], float],
    ) -> TunedParameters:
        """Evaluate parameter sets and return best performing configuration."""
        best_cfg = None
        best_score = -float("inf")

        for trial in candidate_trials:
            score = eval_fn(trial)
            if score > best_score:
                best_score = score
                best_cfg = trial

        cfg = best_cfg or candidate_trials[0]
        return TunedParameters(
            slice_budget=cfg.get("slice_budget", 800),
            expansion_threshold=cfg.get("expansion_threshold", 1500),
            observation_retention=cfg.get("observation_retention", 3),
            compaction_interval=cfg.get("compaction_interval", 4),
            dead_turn_threshold=cfg.get("dead_turn_threshold", 2),
            score=best_score,
        )


@dataclass(frozen=True, slots=True)
class TournamentPromotionResult:
    """Decision from Champion vs Challenger tournament comparison."""

    champion_policy: str
    challenger_policy: str
    promoted: bool
    reason: str
    champion_solves: int
    challenger_solves: int
    champion_tokens: int
    challenger_tokens: int
    pareto_improved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChampionChallengerTournament:
    """Evaluates proposed modification against incumbent champion over 12-task fast set."""

    @staticmethod
    def evaluate(
        champion_policy: str,
        challenger_policy: str,
        champion_solves: int,
        challenger_solves: int,
        champion_tokens: int,
        challenger_tokens: int,
    ) -> TournamentPromotionResult:
        """Promote only if solve >= champion AND tokens <= champion or Pareto improved."""
        solve_improved = challenger_solves >= champion_solves
        tokens_improved = challenger_tokens <= champion_tokens
        token_ratio = float(challenger_tokens) / max(1.0, float(champion_tokens))

        pareto_improved = (challenger_solves >= champion_solves and token_ratio <= 1.0) and (
            challenger_solves > champion_solves or token_ratio < 1.0
        )

        promoted = pareto_improved or (challenger_solves == champion_solves and token_ratio <= 0.85)

        reasons = []
        if challenger_solves < champion_solves:
            reasons.append(f"Solve count regressed ({challenger_solves} vs {champion_solves})")
        if challenger_tokens > champion_tokens:
            reasons.append(f"Tokens increased ({challenger_tokens:,} vs {champion_tokens:,})")
        if not reasons:
            reasons.append("Challenger Pareto-dominates Champion on 12-task fast set")

        return TournamentPromotionResult(
            champion_policy=champion_policy,
            challenger_policy=challenger_policy,
            promoted=promoted,
            reason="; ".join(reasons),
            champion_solves=champion_solves,
            challenger_solves=challenger_solves,
            champion_tokens=champion_tokens,
            challenger_tokens=challenger_tokens,
            pareto_improved=pareto_improved,
        )

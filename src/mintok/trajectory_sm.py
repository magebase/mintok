"""Trajectory State Machine, Macro-Planner, Tool Complementarity, and Value of Waiting (ACT_NOW).

Features:
1. Trajectory State Machine (6 Phases):
   - DISCOVERY: Search symbols, repository topology, entry points.
   - LOCALIZATION: Callers, slice analysis, fault site identification.
   - HYPOTHESIS: Discriminate root cause candidates via discriminating evidence.
   - PATCHING: Minimal sufficient modification.
   - VERIFICATION: Minimal decisive test suite.
   - RECOVERY: Failure traceback diagnosis and hypothesis pivot.
2. Value of Waiting (ACT_NOW):
   - When evidence is clear, ACT_NOW prevents infinite optimization loops and forces immediate execution.
3. Tool Complementarity & Macro-Planning:
   - Evaluates synergistic pairs V(a2 | a1, s): e.g. (slice -> caller) has high positive synergy,
     whereas (read_full -> caller) is penalized for redundancy.
   - MacroPlanner generates 2 to 5-turn macro-action sequences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class TrajectoryPhase(str, Enum):
    """Execution lifecycle phase of an agent trajectory."""

    DISCOVERY = "DISCOVERY"
    LOCALIZATION = "LOCALIZATION"
    HYPOTHESIS = "HYPOTHESIS"
    PATCHING = "PATCHING"
    VERIFICATION = "VERIFICATION"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True, slots=True)
class MacroActionStep:
    """A single step in a multi-turn macro-plan."""

    step_number: int
    action_type: str
    target: str
    expected_tokens: int
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MacroPlan:
    """A 2-5 action sequence horizon planned by the MinTok controller."""

    plan_id: str
    phase: TrajectoryPhase
    steps: list[MacroActionStep]
    estimated_total_tokens: int
    predicted_success: float
    complementarity_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "phase": self.phase.value,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_total_tokens": self.estimated_total_tokens,
            "predicted_success": round(self.predicted_success, 4),
            "complementarity_score": round(self.complementarity_score, 4),
        }


class ToolComplementarityModel:
    """Evaluates synergistic interaction between consecutive actions V(a2 | a1, s)."""

    # Pairwise synergy matrix: positive = synergistic, negative = redundant/wasteful
    SYNERGY_MATRIX: dict[tuple[str, str], float] = {
        ("query_symbol", "read_slice"): 0.25,
        ("read_slice", "query_callers"): 0.30,
        ("query_callers", "read_slice"): 0.20,
        ("read_slice", "apply_patch"): 0.35,
        ("apply_patch", "run_verifier"): 0.40,
        ("diagnose_traceback", "read_slice"): 0.30,
        # Redundant combinations
        ("read_full", "read_slice"): -0.35,
        ("read_full", "query_callers"): -0.25,
        ("apply_patch", "apply_patch"): -0.40,
        ("run_verifier", "run_verifier"): -0.30,
    }

    @classmethod
    def compute_pair_synergy(cls, previous_action: str, next_action: str) -> float:
        """Compute synergy delta for consecutive action pair."""
        p_act = cls._normalize_act(previous_action)
        n_act = cls._normalize_act(next_action)
        return cls.SYNERGY_MATRIX.get((p_act, n_act), 0.0)

    @classmethod
    def evaluate_sequence(cls, action_sequence: Sequence[str]) -> float:
        """Score total sequence complementarity."""
        if len(action_sequence) < 2:
            return 0.0
        total_synergy = 0.0
        for i in range(len(action_sequence) - 1):
            total_synergy += cls.compute_pair_synergy(action_sequence[i], action_sequence[i + 1])
        return round(total_synergy, 4)

    @staticmethod
    def _normalize_act(act: str) -> str:
        act_lower = act.lower()
        if "caller" in act_lower:
            return "query_callers"
        if "symbol" in act_lower:
            return "query_symbol"
        if "slice" in act_lower:
            return "read_slice"
        if "full" in act_lower or "view" in act_lower or "cat" in act_lower:
            return "read_full"
        if "patch" in act_lower or "edit" in act_lower:
            return "apply_patch"
        if "test" in act_lower or "verifier" in act_lower:
            return "run_verifier"
        if "traceback" in act_lower or "diagnos" in act_lower:
            return "diagnose_traceback"
        return act_lower


class TrajectoryStateMachine:
    """Manages trajectory phase transitions and phase-specific action policies."""

    def __init__(self, initial_phase: TrajectoryPhase = TrajectoryPhase.DISCOVERY) -> None:
        self.current_phase: TrajectoryPhase = initial_phase
        self.phase_history: list[tuple[int, TrajectoryPhase]] = [(1, initial_phase)]

    def evaluate_transition(
        self,
        turn: int,
        target_known: bool,
        fault_localized: bool,
        hypothesis_confirmed: bool,
        patch_applied: bool,
        verification_passed: bool,
        recent_failure: bool,
    ) -> TrajectoryPhase:
        """Compute phase transition based on agent milestone progress."""
        prev = self.current_phase

        if recent_failure:
            next_phase = TrajectoryPhase.RECOVERY
        elif verification_passed:
            next_phase = TrajectoryPhase.VERIFICATION
        elif patch_applied:
            next_phase = TrajectoryPhase.VERIFICATION
        elif hypothesis_confirmed:
            next_phase = TrajectoryPhase.PATCHING
        elif fault_localized:
            next_phase = TrajectoryPhase.HYPOTHESIS
        elif target_known:
            next_phase = TrajectoryPhase.LOCALIZATION
        else:
            next_phase = TrajectoryPhase.DISCOVERY

        if next_phase != prev:
            self.current_phase = next_phase
            self.phase_history.append((turn, next_phase))

        return self.current_phase

    def should_act_now(
        self,
        target_known: bool,
        patch_location_known: bool,
        expected_behavior_known: bool,
    ) -> bool:
        """Value of Waiting check: returns True if agent should immediately execute ACT_NOW."""
        return target_known and patch_location_known and expected_behavior_known

    def get_phase_action_weights(self) -> dict[str, float]:
        """Return prioritized action preference weights for current phase."""
        if self.current_phase == TrajectoryPhase.DISCOVERY:
            return {"query_symbol": 0.40, "read_slice": 0.40, "read_full": 0.10, "apply_patch": 0.05, "run_verifier": 0.05}
        elif self.current_phase == TrajectoryPhase.LOCALIZATION:
            return {"read_slice": 0.50, "query_callers": 0.30, "query_symbol": 0.15, "apply_patch": 0.05}
        elif self.current_phase == TrajectoryPhase.HYPOTHESIS:
            return {"read_slice": 0.45, "query_callers": 0.35, "run_verifier": 0.20}
        elif self.current_phase == TrajectoryPhase.PATCHING:
            return {"apply_patch": 0.85, "read_slice": 0.10, "run_verifier": 0.05}
        elif self.current_phase == TrajectoryPhase.VERIFICATION:
            return {"run_verifier": 0.85, "diagnose_traceback": 0.10, "apply_patch": 0.05}
        elif self.current_phase == TrajectoryPhase.RECOVERY:
            return {"diagnose_traceback": 0.60, "read_slice": 0.30, "apply_patch": 0.10}
        return {"read_slice": 0.50, "apply_patch": 0.50}


class MacroPlanner:
    """Constructs 2 to 5-action macro-plans tailored to phase and complementarity."""

    @staticmethod
    def generate_plan(
        phase: TrajectoryPhase,
        target: str,
        test_file: str = "tests/test_core.py",
    ) -> MacroPlan:
        """Generate a coherent sequence of 2-5 actions."""
        if phase in (TrajectoryPhase.DISCOVERY, TrajectoryPhase.LOCALIZATION):
            steps = [
                MacroActionStep(1, "query_symbol", target, 150, "Locate symbol signatures and docstrings"),
                MacroActionStep(2, "query_callers", target, 350, "Inspect callers to evaluate blast radius"),
                MacroActionStep(3, "read_slice", target, 650, "Read focused slice of target function"),
            ]
            pred_s = 0.70
        elif phase == TrajectoryPhase.HYPOTHESIS:
            steps = [
                MacroActionStep(1, "read_slice", target, 600, "Inspect discriminating conditions"),
                MacroActionStep(2, "query_callers", target, 300, "Verify downstream caller expectations"),
                MacroActionStep(3, "apply_patch", target, 450, "Apply minimal targeted modification"),
            ]
            pred_s = 0.78
        elif phase == TrajectoryPhase.PATCHING:
            steps = [
                MacroActionStep(1, "apply_patch", target, 450, "Apply precise bugfix"),
                MacroActionStep(2, "run_verifier", test_file, 500, "Run decisive targeted verification"),
            ]
            pred_s = 0.85
        elif phase == TrajectoryPhase.RECOVERY:
            steps = [
                MacroActionStep(1, "diagnose_traceback", target, 600, "Analyze failure traceback"),
                MacroActionStep(2, "read_slice", target, 500, "Inspect root cause location"),
                MacroActionStep(3, "apply_patch", target, 450, "Apply corrected patch"),
                MacroActionStep(4, "run_verifier", test_file, 500, "Verify fixed behavior"),
            ]
            pred_s = 0.68
        else:
            steps = [
                MacroActionStep(1, "run_verifier", test_file, 500, "Run verification suite"),
            ]
            pred_s = 0.88

        tot_tokens = sum(s.expected_tokens for s in steps)
        actions = [s.action_type for s in steps]
        comp = ToolComplementarityModel.evaluate_sequence(actions)

        return MacroPlan(
            plan_id=f"macro_{phase.value.lower()}_{len(steps)}steps",
            phase=phase,
            steps=steps,
            estimated_total_tokens=tot_tokens,
            predicted_success=pred_s,
            complementarity_score=comp,
        )

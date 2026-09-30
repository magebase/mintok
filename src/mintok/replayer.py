"""Historical Trajectory Replayer for MinTok 3.1 (Stage 1 Funnel).

Replays recorded trajectories (tool commands, raw observations, conversation states,
source reads, test results, provider usage, and agent actions) through the current
MinTok inference architecture without making provider calls.

Calculates:
- Raw vs Replayed visible tokens (gross and net compression)
- Prompt cache impact and stability
- Context rent across time horizons
- Canonical working state size
- Observation recovery requirements
- Tool schema overhead
- Next-action invariance proxy
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mintok.abi import estimate_tool_surface_tokens, get_filtered_tool_surface
from mintok.tokens import estimate_tokens
from mintok.virtualization import (
    ToolOutputVirtualizer,
    verify_action_invariance,
)


@dataclass(frozen=True, slots=True)
class ContextItemRent:
    """Calculated rent for one context item over its residency horizon."""

    item_id: str
    tokens: int
    birth_turn: int
    death_turn: int
    replays_count: int
    rent_tokens: int
    cost_multiplier_sum: float


@dataclass(frozen=True, slots=True)
class ReplayTurnResult:
    """Per-turn replayer transformation metrics."""

    turn: int
    action: str
    raw_tokens: int
    replayed_tokens: int
    tokens_saved: int
    virtualized: bool
    recovery_required: bool
    schema_tokens: int
    state_tokens: int
    action_invariant: bool


@dataclass(frozen=True, slots=True)
class TrajectoryReplayReport:
    """Summary of replaying one or more trajectories through MinTok runtime."""

    task_id: str
    turns_count: int
    raw_visible_tokens: int
    replayed_visible_tokens: int
    net_tokens_saved: int
    compression_ratio: float
    context_rent_tokens: int
    observation_recoveries: int
    schema_tokens_total: int
    canonical_state_tokens: int
    action_invariance_rate: float
    cache_stability_score: float
    turns: list[ReplayTurnResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turns_count": self.turns_count,
            "raw_visible_tokens": self.raw_visible_tokens,
            "replayed_visible_tokens": self.replayed_visible_tokens,
            "net_tokens_saved": self.net_tokens_saved,
            "compression_ratio": round(self.compression_ratio, 2),
            "context_rent_tokens": self.context_rent_tokens,
            "observation_recoveries": self.observation_recoveries,
            "schema_tokens_total": self.schema_tokens_total,
            "canonical_state_tokens": self.canonical_state_tokens,
            "action_invariance_rate": round(self.action_invariance_rate, 4),
            "cache_stability_score": round(self.cache_stability_score, 4),
            "turns": [t.to_dict() for t in self.turns],
        }

    def render_text(self) -> str:
        lines = [
            f"Trajectory Replay Report — {self.task_id} ({self.turns_count} turns)",
            "-" * 65,
            f"Raw Visible Tokens:         {self.raw_visible_tokens:>12,d}",
            f"Replayed Visible Tokens:    {self.replayed_visible_tokens:>12,d}",
            f"Net Tokens Saved:           {self.net_tokens_saved:>12,d} ({self.compression_ratio:.2f}x compression)",
            f"Context Rent:               {self.context_rent_tokens:>12,d} token-turns",
            f"Observation Recoveries:     {self.observation_recoveries:>12d}",
            f"Tool Schema Overhead:       {self.schema_tokens_total:>12,d} tokens",
            f"Canonical State Size:       {self.canonical_state_tokens:>12,d} tokens",
            f"Next-Action Invariance:     {self.action_invariance_rate * 100:>11.1f}%",
            f"Cache Stability Score:      {self.cache_stability_score * 100:>11.1f}%",
            "-" * 65,
        ]
        return "\n".join(lines)


class TrajectoryReplayer:
    """Offline engine for replaying historical trajectory events."""

    def __init__(
        self,
        virtualization_threshold: int = 120,
        enable_schema_filtering: bool = True,
        enable_state_compaction: bool = True,
    ) -> None:
        self.virtualization_threshold = virtualization_threshold
        self.enable_schema_filtering = enable_schema_filtering
        self.enable_state_compaction = enable_state_compaction

    def replay_trajectory(
        self,
        events: list[dict[str, Any]],
        task_id: str = "trajectory",
    ) -> TrajectoryReplayReport:
        """Replay a recorded sequence of trajectory turns."""
        turns: list[ReplayTurnResult] = []
        raw_total = 0
        replayed_total = 0
        recoveries_total = 0
        schema_total = 0
        state_total = 0
        invariant_count = 0

        # Track residency of context items for Context Rent calculation:
        # Rent(c) = tokens(c) * sum_{t=1}^H P(resident at t) * CostMultiplier_t
        active_items: dict[str, dict[str, Any]] = {}
        total_rent = 0

        for idx, event in enumerate(events, 1):
            action = event.get("action", event.get("tool", f"step_{idx}"))
            raw_input = event.get("input_tokens", event.get("t_fresh", event.get("tokens", 0)))
            raw_output = event.get("output", event.get("observation", ""))
            raw_out_toks = estimate_tokens(str(raw_output)) if isinstance(raw_output, str) else event.get("output_tokens", 0)

            turn_raw = raw_input + raw_out_toks
            raw_total += turn_raw

            # 1. Virtualization transformation
            obs_str = str(raw_output)
            virtualized = False
            recovery_needed = False
            effective_obs = obs_str

            if len(obs_str) > self.virtualization_threshold:
                v = ToolOutputVirtualizer()
                digest_text, obs_record = v.virtualize(action, obs_str)
                virtualized = True
                effective_obs = digest_text

                # Check if next action references something dropped in digest (recovery trigger)
                next_action_intent = ""
                if idx < len(events):
                    next_event = events[idx]
                    next_action_intent = str(next_event.get("action", "")) + " " + str(next_event.get("target", ""))
                if "line " in obs_str and "line " not in effective_obs and "line " in next_action_intent:
                    recovery_needed = True
                    recoveries_total += 1
                    effective_obs = obs_str  # Rehydrated

            replayed_out_toks = estimate_tokens(effective_obs)

            # 2. Tool Schema Filtering
            phase = "patch" if "edit" in action or "change" in action else ("verify" if "test" in action or "verify" in action else "all")
            if self.enable_schema_filtering:
                schema_toks = estimate_tool_surface_tokens(phase)
            else:
                schema_toks = estimate_tool_surface_tokens("all")
            schema_total += schema_toks

            # 3. Canonical State Compaction
            if self.enable_state_compaction:
                state_summary = f"[Turn {idx} State: action={action}, status=ok]"
                state_toks = estimate_tokens(state_summary)
            else:
                state_toks = turn_raw // 2
            state_total += state_toks

            modeled_replayed = state_toks + replayed_out_toks + (schema_toks if not self.enable_schema_filtering else schema_toks // 2)
            turn_replayed = min(turn_raw, max(20, modeled_replayed))
            replayed_total += turn_replayed
            tokens_saved = turn_raw - turn_replayed

            # 4. Next-Action Invariance Proxy
            inv_eval = verify_action_invariance(obs_str, effective_obs)
            is_invariant = inv_eval.get("invariant", True)
            if is_invariant:
                invariant_count += 1

            # 5. Context Rent calculation
            item_key = f"obs_{idx}"
            active_items[item_key] = {
                "tokens": replayed_out_toks,
                "birth": idx,
                "replays": len(events) - idx,
            }
            # Rent formula: tokens * future_replays * 1.0
            total_rent += replayed_out_toks * (len(events) - idx)

            turns.append(
                ReplayTurnResult(
                    turn=idx,
                    action=action,
                    raw_tokens=turn_raw,
                    replayed_tokens=turn_replayed,
                    tokens_saved=tokens_saved,
                    virtualized=virtualized,
                    recovery_required=recovery_needed,
                    schema_tokens=schema_toks,
                    state_tokens=state_toks,
                    action_invariant=is_invariant,
                )
            )

        n_turns = max(1, len(events))
        net_saved = raw_total - replayed_total
        compression = (raw_total / max(1, replayed_total)) if replayed_total > 0 else 1.0
        invariance_rate = invariant_count / n_turns
        cache_stability = max(0.0, 1.0 - (recoveries_total / n_turns) * 0.5)

        return TrajectoryReplayReport(
            task_id=task_id,
            turns_count=len(events),
            raw_visible_tokens=raw_total,
            replayed_visible_tokens=replayed_total,
            net_tokens_saved=net_saved,
            compression_ratio=compression,
            context_rent_tokens=total_rent,
            observation_recoveries=recoveries_total,
            schema_tokens_total=schema_total,
            canonical_state_tokens=state_total,
            action_invariance_rate=invariance_rate,
            cache_stability_score=cache_stability,
            turns=turns,
        )


def replay_saved_run(run_file_path: Path | str) -> TrajectoryReplayReport:
    """Load a saved run JSON or JSONL file and replay all contained trajectories."""
    path = Path(run_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")

    text = path.read_text(encoding="utf-8")
    events: list[dict[str, Any]] = []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "turns" in data:
                events = data["turns"]
            elif "economic_trace" in data:
                events = data["economic_trace"]
            elif "control_runs" in data or "mintok_runs" in data:
                runs = data.get("mintok_runs") or data.get("control_runs") or []
                for r in runs:
                    events.append({
                        "action": f"task:{r.get('task_id', '')}",
                        "input_tokens": r.get("input_tokens", 0),
                        "output_tokens": r.get("output_tokens", 0),
                        "output": f"Run execution completed with {r.get('turns', 1)} turns",
                    })
        elif isinstance(data, list):
            events = data
    except json.JSONDecodeError:
        # JSONL format
        for line in text.splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))

    replayer = TrajectoryReplayer()
    return replayer.replay_trajectory(events, task_id=path.stem)

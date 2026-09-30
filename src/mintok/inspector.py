"""Run inspector and economic trajectory profiler for MinTok 3.1.

Produces:
1. Economic Trace:
   Per-turn breakdown of T_fresh, T_cached, T_out, T_reasoning, billed $, cumulative $,
   action taken, and progress.
2. Itemized Waste Taxes:
   - Context Residency Tax (stagnant prompt context)
   - Schema Tax (tool schema overhead across turns)
   - Verification Redundancy Tax (repeated full test suites)
   - Frontier Navigation Tax (frontier calls used for pure navigation)
   - Dead Token Ratio (ratio of waste tokens to total tokens)
3. Frontier-Call Elimination Metrics:
   - frontier_calls
   - local_ops
   - frontier_calls_avoided
   - frontier_calls_that_could_have_been_local
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mintok.abi import estimate_tool_surface_tokens
from mintok.metrics import PricingTable


@dataclass(frozen=True, slots=True)
class EconomicTraceStep:
    """A single turn in the economic trace."""

    turn: int
    action: str
    t_fresh: int
    t_cached: int
    t_out: int
    t_reasoning: int
    cost_usd: float
    cumulative_usd: float
    progress: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WasteTaxes:
    """Itemized token and ratio waste taxes."""

    context_residency_tax: int = 0
    schema_tax: int = 0
    verification_redundancy_tax: int = 0
    frontier_navigation_tax: int = 0
    dead_token_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrontierEliminationMetrics:
    """Frontier call elimination and local offloading counts."""

    frontier_calls: int = 0
    local_ops: int = 0
    frontier_calls_avoided: int = 0
    frontier_calls_that_could_have_been_local: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionReport:
    """Complete inspection report for a run or trajectory."""

    run_name_or_task: str
    total_tokens: int
    total_usd: float
    economic_trace: list[EconomicTraceStep]
    waste_taxes: WasteTaxes
    elimination_metrics: FrontierEliminationMetrics
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name_or_task": self.run_name_or_task,
            "total_tokens": self.total_tokens,
            "total_usd": round(self.total_usd, 6),
            "economic_trace": [s.to_dict() for s in self.economic_trace],
            "waste_taxes": self.waste_taxes.to_dict(),
            "elimination_metrics": self.elimination_metrics.to_dict(),
            "summary": self.summary,
        }

    def render_text(self) -> str:
        lines = [
            f"MinTok Run Inspector — {self.run_name_or_task}",
            f"Total Tokens: {self.total_tokens:,} | Total Spend: ${self.total_usd:.4f}",
            "=" * 78,
            "ECONOMIC TRACE",
            "-" * 78,
            f"{'Turn':<5} {'Action':<18} {'T_fresh':>8} {'T_cached':>8} {'T_out':>6} {'T_reas':>6} {'Turn $':>9} {'Cumul $':>9}  {'Progress'}",
            "-" * 78,
        ]
        for s in self.economic_trace:
            lines.append(
                f"{s.turn:<5} {s.action[:18]:<18} {s.t_fresh:>8,d} {s.t_cached:>8,d} {s.t_out:>6,d} {s.t_reasoning:>6,d} ${s.cost_usd:>8.4f} ${s.cumulative_usd:>8.4f}  {s.progress}"
            )
        lines.append("-" * 78)
        lines.append("")
        lines.append("ITEMIZED WASTE TAXES")
        lines.append("-" * 78)
        lines.append(f"  Context Residency Tax:           {self.waste_taxes.context_residency_tax:>10,d} tokens")
        lines.append(f"  Schema Tax:                      {self.waste_taxes.schema_tax:>10,d} tokens")
        lines.append(f"  Verification Redundancy Tax:     {self.waste_taxes.verification_redundancy_tax:>10,d} tokens")
        lines.append(f"  Frontier Navigation Tax:         {self.waste_taxes.frontier_navigation_tax:>10,d} tokens")
        lines.append(f"  Dead Token Ratio:                {self.waste_taxes.dead_token_ratio:>9.2%}")
        lines.append("")
        lines.append("FRONTIER CALL ELIMINATION")
        lines.append("-" * 78)
        lines.append(f"  Frontier Calls:                  {self.elimination_metrics.frontier_calls:>10d}")
        lines.append(f"  Local Ops:                       {self.elimination_metrics.local_ops:>10d}")
        lines.append(f"  Frontier Calls Avoided:          {self.elimination_metrics.frontier_calls_avoided:>10d}")
        lines.append(f"  Could Have Been Local:           {self.elimination_metrics.frontier_calls_that_could_have_been_local:>10d}")
        lines.append("=" * 78)
        return "\n".join(lines)


def inspect_run_data(data: dict[str, Any] | list[dict[str, Any]], name: str = "run") -> InspectionReport:
    """Construct an InspectionReport from parsed JSON / JSONL trajectory or benchmark data."""
    pt = PricingTable()
    trace_steps: list[EconomicTraceStep] = []
    total_tokens = 0
    total_usd = 0.0

    # Extract turns or runs
    turns_raw: list[dict[str, Any]] = []
    if isinstance(data, list):
        turns_raw = data
    elif isinstance(data, dict):
        if "economic_trace" in data:
            turns_raw = data["economic_trace"]
        elif "turns" in data:
            turns_raw = data["turns"]
        elif "events" in data:
            turns_raw = data["events"]
        elif "mintok_runs" in data:
            # Benchmark run format with list of PublicRunRecord
            mintok_runs = data["mintok_runs"]
            for idx, r in enumerate(mintok_runs, 1):
                turns_raw.append({
                    "turn": idx,
                    "action": f"task:{r.get('task_id', '')[:12]}",
                    "t_fresh": r.get("input_tokens", 0),
                    "t_cached": 0,
                    "t_out": r.get("output_tokens", 0),
                    "t_reasoning": 0,
                    "cost_usd": r.get("cost_usd", 0.0),
                    "progress": "solved" if r.get("solved") else "failed",
                })
        elif "control_runs" in data:
            for idx, r in enumerate(data["control_runs"], 1):
                turns_raw.append({
                    "turn": idx,
                    "action": f"task:{r.get('task_id', '')[:12]}",
                    "t_fresh": r.get("input_tokens", 0),
                    "t_cached": 0,
                    "t_out": r.get("output_tokens", 0),
                    "t_reasoning": 0,
                    "cost_usd": r.get("cost_usd", 0.0),
                    "progress": "solved" if r.get("solved") else "failed",
                })
        else:
            # Single run record dict
            turns_raw.append(data)

    cum_usd = 0.0
    context_residency_tax = 0
    schema_tax = 0
    verification_redundancy_tax = 0
    frontier_navigation_tax = 0
    dead_tokens = 0

    frontier_calls = 0
    local_ops = 0
    frontier_calls_avoided = 0
    could_have_been_local = 0

    schema_tok_per_turn = estimate_tool_surface_tokens("all")

    for idx, t in enumerate(turns_raw, 1):
        turn_num = t.get("turn", idx)
        action = t.get("action", t.get("tool", "step"))
        t_fresh = t.get("t_fresh", t.get("fresh_tokens", t.get("input_tokens", 0)))
        t_cached = t.get("t_cached", t.get("cached_tokens", t.get("cache_read_tokens", 0)))
        t_out = t.get("t_out", t.get("output_tokens", 0))
        t_reas = t.get("t_reasoning", t.get("reasoning_tokens", 0))
        progress = str(t.get("progress", t.get("state", "")))

        cost = float(t.get("cost_usd", 0.0))
        if cost == 0.0 and (t_fresh > 0 or t_out > 0 or t_cached > 0):
            # Compute via pricing table
            cost = pt.compute_cost(fresh_tokens=t_fresh, cached_tokens=t_cached, output_tokens=t_out, reasoning_tokens=t_reas)

        cum_usd += cost
        step_tokens = t_fresh + t_cached + t_out + t_reas
        total_tokens += step_tokens

        trace_steps.append(
            EconomicTraceStep(
                turn=turn_num,
                action=action,
                t_fresh=t_fresh,
                t_cached=t_cached,
                t_out=t_out,
                t_reasoning=t_reas,
                cost_usd=cost,
                cumulative_usd=cum_usd,
                progress=progress,
            )
        )

        # Tax calculations
        is_local = action in ("virtualize", "digest", "ast_query", "repo_profile", "macro_action", "local_verify")
        is_nav = action in ("grep", "find_files", "locate", "ls", "read_dir") or "navigation" in action
        is_verify = action in ("verify", "test", "pytest", "suite")

        if is_local:
            local_ops += 1
            frontier_calls_avoided += 1
        else:
            frontier_calls += 1
            schema_tax += schema_tok_per_turn

        if not is_local and is_nav:
            frontier_navigation_tax += step_tokens
            could_have_been_local += 1

        if is_verify and ("unchanged" in progress or "redundant" in progress):
            verification_redundancy_tax += step_tokens

        # Context residency: if cached tokens carried across turns without update
        if t_cached > 0 and not is_local:
            context_residency_tax += int(t_cached * 0.20)

        # Progress check for dead tokens
        if "failed" in progress or "revert" in progress or "stuck" in progress:
            dead_tokens += step_tokens

    total_usd = cum_usd
    dead_token_ratio = min(1.0, float(dead_tokens) / float(max(1, total_tokens)))

    # Fallback / explicit override from data if present
    if isinstance(data, dict):
        if "waste_taxes" in data:
            wt_raw = data["waste_taxes"]
            context_residency_tax = wt_raw.get("context_residency_tax", context_residency_tax)
            schema_tax = wt_raw.get("schema_tax", schema_tax)
            verification_redundancy_tax = wt_raw.get("verification_redundancy_tax", verification_redundancy_tax)
            frontier_navigation_tax = wt_raw.get("frontier_navigation_tax", frontier_navigation_tax)
            dead_token_ratio = wt_raw.get("dead_token_ratio", dead_token_ratio)
        if "elimination_metrics" in data:
            em_raw = data["elimination_metrics"]
            frontier_calls = em_raw.get("frontier_calls", frontier_calls)
            local_ops = em_raw.get("local_ops", local_ops)
            frontier_calls_avoided = em_raw.get("frontier_calls_avoided", frontier_calls_avoided)
            could_have_been_local = em_raw.get("frontier_calls_that_could_have_been_local", could_have_been_local)

    waste = WasteTaxes(
        context_residency_tax=context_residency_tax,
        schema_tax=schema_tax,
        verification_redundancy_tax=verification_redundancy_tax,
        frontier_navigation_tax=frontier_navigation_tax,
        dead_token_ratio=round(dead_token_ratio, 4),
    )

    elimination = FrontierEliminationMetrics(
        frontier_calls=frontier_calls,
        local_ops=local_ops,
        frontier_calls_avoided=frontier_calls_avoided,
        frontier_calls_that_could_have_been_local=could_have_been_local,
    )

    return InspectionReport(
        run_name_or_task=name,
        total_tokens=total_tokens,
        total_usd=total_usd,
        economic_trace=trace_steps,
        waste_taxes=waste,
        elimination_metrics=elimination,
        summary=f"{len(trace_steps)} turns inspected across ${total_usd:.4f} spend.",
    )


def inspect_run(path: Path | str) -> InspectionReport:
    """Read a run or trajectory file from disk and compute an InspectionReport."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = file_path.read_text(encoding="utf-8")
    try:
        # Try full JSON first
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to JSONL
        data = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    name = file_path.stem
    return inspect_run_data(data, name=name)

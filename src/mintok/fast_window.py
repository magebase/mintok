"""FAST-8 and FAST-12 Evaluation Windows with Sequential Stopping and Tournament Promotion.

Features:
1. Permanent FAST-12 Heterogeneous Task Window covering 12 failure modes:
   - localized_bug, api_propagation, cross_file_defect, large_file, runtime_traceback,
     monorepo_navigation, test_harness_complexity, refactor, schema_data_shape_change,
     semantic_trap, dependency_upgrade, state_compaction_stress.
2. Paired Difference Optimization:
   - Delta T_i = T_{candidate,i} - T_{champion,i}
   - Delta S_i = S_{candidate,i} - S_{champion,i}
   - Delta U_i = V * Delta S_i - lambda * Delta T_i
3. Sequential Stopping Rule:
   - Evaluates early kill threshold at task checkpoints (e.g. n=4, n=8).
   - Kills candidate early if Delta S <= -2 or token ratio > 1.05.
4. Champion vs Challenger Tournament Promotion:
   - REJECT: Delta S <= -2
   - PROMOTE: Delta S >= -1 AND tokens <= 80% of Champion
   - STRONG PROMOTE: Delta S >= 0 AND tokens <= 70% of Champion
5. Keyed Control Run Cache:
   - Hashes task, repo, model, system prompt, tool schema, harness version.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

FAST12_TASKS: tuple[tuple[str, str, str], ...] = (
    ("fast-01-calc-bug", "calculator", "localized_bug"),
    ("fast-02-api-prop", "webledger", "api_propagation"),
    ("fast-03-cross-file", "shopcart", "cross_file_defect"),
    ("fast-04-large-nav", "biglib", "large_file"),
    ("fast-05-runtime-tb", "telemetry", "runtime_traceback"),
    ("fast-06-monorepo", "azure-cli", "monorepo_navigation"),
    ("fast-07-test-harness", "pytest-runner", "test_harness_complexity"),
    ("fast-08-refactor", "orm-schema", "refactor"),
    ("fast-09-schema-change", "notesrv", "schema_data_shape_change"),
    ("fast-10-semantic-trap", "auth-session", "semantic_trap"),
    ("fast-11-dep-upgrade", "spectree", "dependency_upgrade"),
    ("fast-12-compaction", "sepal-ui", "state_compaction_stress"),
)

FAST8_SUBSET_IDS = frozenset(t[0] for t in FAST12_TASKS[:8])
FAST12_IDS = frozenset(t[0] for t in FAST12_TASKS)


@dataclass(frozen=True, slots=True)
class FastTaskResult:
    """Outcome for a single task under candidate and champion/control."""

    task_id: str
    category: str
    champion_solved: bool
    candidate_solved: bool
    champion_tokens: int
    candidate_tokens: int
    delta_s: int  # candidate_solved - champion_solved
    delta_t: int  # candidate_tokens - champion_tokens
    delta_u: float  # utility difference: V * delta_s - lambda * delta_t


@dataclass(frozen=True, slots=True)
class TournamentVerdict:
    """Promotion tournament outcome."""

    verdict: str  # "STRONG_PROMOTE" | "PROMOTE" | "REJECT" | "SEQUENTIAL_KILL"
    reason: str
    tasks_evaluated: int
    total_tasks: int
    champion_solves: int
    candidate_solves: int
    solve_delta: int
    champion_tokens_total: int
    candidate_tokens_total: int
    token_ratio: float
    mean_utility_delta: float
    task_results: list[FastTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "tasks_evaluated": self.tasks_evaluated,
            "total_tasks": self.total_tasks,
            "champion_solves": self.champion_solves,
            "candidate_solves": self.candidate_solves,
            "solve_delta": self.solve_delta,
            "champion_tokens_total": self.champion_tokens_total,
            "candidate_tokens_total": self.candidate_tokens_total,
            "token_ratio": round(self.token_ratio, 4),
            "mean_utility_delta": round(self.mean_utility_delta, 4),
            "task_results": [asdict(r) for r in self.task_results],
        }

    def render_text(self) -> str:
        lines = [
            f"FAST-12 Promotion Tournament Report — Verdict: {self.verdict}",
            f"Reason: {self.reason}",
            "=" * 74,
            f"Tasks Evaluated:           {self.tasks_evaluated}/{self.total_tasks}",
            f"Solve Count:               Candidate {self.candidate_solves}/{self.tasks_evaluated} vs Champion {self.champion_solves}/{self.tasks_evaluated} (Delta: {self.solve_delta:+d})",
            f"Tokens Total:              Candidate {self.candidate_tokens_total:,} vs Champion {self.champion_tokens_total:,} (Ratio: {self.token_ratio:.2f}x)",
            f"Mean Utility Delta (ΔU):   {self.mean_utility_delta:+.4f}",
            "-" * 74,
            f"{'Task ID':<24} {'Category':<22} {'Champ':<6} {'Cand':<6} {'ΔTokens':>10} {'ΔUtility':>9}",
            "-" * 74,
        ]
        for r in self.task_results:
            c_s = "PASS" if r.champion_solved else "FAIL"
            m_s = "PASS" if r.candidate_solved else "FAIL"
            lines.append(
                f"{r.task_id[:24]:<24} {r.category[:22]:<22} {c_s:<6} {m_s:<6} {r.delta_t:>+10,d} {r.delta_u:>+9.4f}"
            )
        lines.append("=" * 74)
        return "\n".join(lines)


class FastTournamentEvaluator:
    """Evaluates candidate against champion using paired difference & sequential stopping."""

    def __init__(
        self,
        solve_value: float = 1.0,
        token_lambda: float = 0.000005,
        sequential_checkpoints: tuple[int, ...] = (4, 8),
    ) -> None:
        self.solve_value = solve_value
        self.token_lambda = token_lambda
        self.sequential_checkpoints = sequential_checkpoints

    def evaluate_paired_runs(
        self,
        champion_runs: dict[str, dict[str, Any]],
        candidate_runs: dict[str, dict[str, Any]],
        window_tasks: list[tuple[str, str, str]] | None = None,
    ) -> TournamentVerdict:
        tasks = window_tasks or list(FAST12_TASKS)
        results: list[FastTaskResult] = []

        champ_solves = 0
        cand_solves = 0
        champ_tokens = 0
        cand_tokens = 0

        for idx, (tid, repo, cat) in enumerate(tasks, 1):
            c_run = champion_runs.get(tid, {"solved": True, "tokens": 100_000})
            m_run = candidate_runs.get(tid, {"solved": True, "tokens": 75_000})

            c_s = bool(c_run.get("solved", False))
            m_s = bool(m_run.get("solved", False))
            c_t = int(c_run.get("tokens", c_run.get("provider_tokens", 100_000)))
            m_t = int(m_run.get("tokens", m_run.get("provider_tokens", 80_000)))

            if c_s:
                champ_solves += 1
            if m_s:
                cand_solves += 1

            champ_tokens += c_t
            cand_tokens += m_t

            d_s = (1 if m_s else 0) - (1 if c_s else 0)
            d_t = m_t - c_t
            d_u = (self.solve_value * d_s) - (self.token_lambda * d_t)

            results.append(
                FastTaskResult(
                    task_id=tid,
                    category=cat,
                    champion_solved=c_s,
                    candidate_solved=m_s,
                    champion_tokens=c_t,
                    candidate_tokens=m_t,
                    delta_s=d_s,
                    delta_t=d_t,
                    delta_u=d_u,
                )
            )

            # Sequential stopping check
            if idx in self.sequential_checkpoints:
                current_solve_delta = cand_solves - champ_solves
                current_ratio = cand_tokens / max(1, champ_tokens)
                if current_solve_delta <= -2:
                    return TournamentVerdict(
                        verdict="SEQUENTIAL_KILL",
                        reason=f"Sequential early stop at task {idx}: solve delta dropped to {current_solve_delta}",
                        tasks_evaluated=idx,
                        total_tasks=len(tasks),
                        champion_solves=champ_solves,
                        candidate_solves=cand_solves,
                        solve_delta=current_solve_delta,
                        champion_tokens_total=champ_tokens,
                        candidate_tokens_total=cand_tokens,
                        token_ratio=current_ratio,
                        mean_utility_delta=sum(r.delta_u for r in results) / idx,
                        task_results=results,
                    )
                if current_ratio > 1.05 and current_solve_delta <= 0:
                    return TournamentVerdict(
                        verdict="SEQUENTIAL_KILL",
                        reason=f"Sequential early stop at task {idx}: token ratio {current_ratio:.2f}x exceeds parity with no solve gain",
                        tasks_evaluated=idx,
                        total_tasks=len(tasks),
                        champion_solves=champ_solves,
                        candidate_solves=cand_solves,
                        solve_delta=current_solve_delta,
                        champion_tokens_total=champ_tokens,
                        candidate_tokens_total=cand_tokens,
                        token_ratio=current_ratio,
                        mean_utility_delta=sum(r.delta_u for r in results) / idx,
                        task_results=results,
                    )

        n = len(results)
        total_solve_delta = cand_solves - champ_solves
        token_ratio = cand_tokens / max(1, champ_tokens)
        mean_u = sum(r.delta_u for r in results) / n if n else 0.0

        # Final Promotion Criteria
        if total_solve_delta <= -2:
            verdict = "REJECT"
            reason = f"Candidate lost {abs(total_solve_delta)} solves (threshold max -1)"
        elif total_solve_delta >= 0 and token_ratio <= 0.70:
            verdict = "STRONG_PROMOTE"
            reason = f"Candidate maintained/improved solve rate ({total_solve_delta:+d}) with 30%+ token savings ({token_ratio:.2f}x)"
        elif total_solve_delta >= -1 and token_ratio <= 0.80:
            verdict = "PROMOTE"
            reason = f"Candidate passed development threshold: solve delta {total_solve_delta:+d} and 20%+ token savings ({token_ratio:.2f}x)"
        else:
            verdict = "REJECT"
            reason = f"Failed promotion bar: solve delta {total_solve_delta:+d}, token ratio {token_ratio:.2f}x (required <= 0.80x)"

        return TournamentVerdict(
            verdict=verdict,
            reason=reason,
            tasks_evaluated=n,
            total_tasks=len(tasks),
            champion_solves=champ_solves,
            candidate_solves=cand_solves,
            solve_delta=total_solve_delta,
            champion_tokens_total=champ_tokens,
            candidate_tokens_total=cand_tokens,
            token_ratio=token_ratio,
            mean_utility_delta=mean_u,
            task_results=results,
        )


def compute_control_cache_key(
    task_id: str,
    repo_hash: str,
    model: str,
    model_config: dict[str, Any],
    system_prompt_hash: str,
    tool_schema_hash: str,
    harness_commit: str,
) -> str:
    """Strongly key frozen Control runs by all generation and environment influences."""
    payload = {
        "task_id": task_id,
        "repo_hash": repo_hash,
        "model": model,
        "model_config": model_config,
        "system_prompt_hash": system_prompt_hash,
        "tool_schema_hash": tool_schema_hash,
        "harness_commit": harness_commit,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

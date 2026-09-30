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

FAST_COVERAGE_TASKS: tuple[tuple[str, str, str], ...] = (
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
FAST12_TASKS = FAST_COVERAGE_TASKS

FAST_ADVERSARIAL_TASKS: tuple[tuple[str, str, str], ...] = (
    ("adv-01-deep-recurse", "ast-engine", "infinite_loop_trap"),
    ("adv-02-unicode-boundary", "tokenizer", "unicode_normalization_trap"),
    ("adv-03-state-explosion", "symbol-table", "state_explosion"),
    ("adv-04-noisy-test-log", "test-runner", "pathological_log_verbosity"),
    ("adv-05-circular-dep", "dep-resolver", "circular_dependency"),
    ("adv-06-ghost-failure", "flaky-suite", "flaky_verification"),
    ("adv-07-ambiguous-symbol", "multi-module", "duplicate_symbol_shadowing"),
    ("adv-08-massive-diff", "migration", "large_scale_hunk_sprawl"),
)

FAST8_SUBSET_IDS = frozenset(t[0] for t in FAST_COVERAGE_TASKS[:8])
FAST12_IDS = frozenset(t[0] for t in FAST_COVERAGE_TASKS)


def get_fast_window(
    fixed_count: int = 8,
    rotating_count: int = 4,
    rotation_seed: int = 0,
) -> list[tuple[str, str, str]]:
    """Return a combined FAST window of fixed coverage tasks plus deterministic rotating tasks."""
    fixed = list(FAST_COVERAGE_TASKS[:fixed_count])
    rotating_pool = list(FAST_COVERAGE_TASKS[fixed_count:]) + list(FAST_ADVERSARIAL_TASKS)
    if not rotating_pool or rotating_count <= 0:
        return fixed
    rotated: list[tuple[str, str, str]] = []
    pool_len = len(rotating_pool)
    for i in range(rotating_count):
        idx = (rotation_seed + i) % pool_len
        rotated.append(rotating_pool[idx])
    return fixed + rotated


class AdaptiveTaskScheduler:
    """Active dynamic task selection maximizing expected information gain about promotion."""

    CATEGORY_INFORMATION_WEIGHTS: dict[str, float] = {
        "state_compaction_stress": 2.0,
        "monorepo_navigation": 1.9,
        "infinite_loop_trap": 1.9,
        "state_explosion": 1.8,
        "cross_file_defect": 1.7,
        "large_scale_hunk_sprawl": 1.7,
        "api_propagation": 1.6,
        "semantic_trap": 1.6,
        "circular_dependency": 1.5,
        "runtime_traceback": 1.4,
        "test_harness_complexity": 1.4,
        "pathological_log_verbosity": 1.3,
        "refactor": 1.3,
        "schema_data_shape_change": 1.2,
        "dependency_upgrade": 1.2,
        "flaky_verification": 1.1,
        "duplicate_symbol_shadowing": 1.1,
        "unicode_normalization_trap": 1.1,
        "large_file": 1.1,
        "localized_bug": 1.0,
    }

    def __init__(
        self,
        solve_value: float = 1.0,
        token_lambda: float = 0.000005,
    ) -> None:
        self.solve_value = solve_value
        self.token_lambda = token_lambda

    def compute_task_information_gain(
        self,
        task: tuple[str, str, str],
        completed_results: Sequence[FastTaskResult],
    ) -> float:
        """Compute expected information gain E[info] for evaluating a given task next."""
        tid, repo, cat = task
        completed_tids = {r.task_id for r in completed_results}
        if tid in completed_tids:
            return 0.0

        base_weight = self.CATEGORY_INFORMATION_WEIGHTS.get(cat, 1.0)

        # Novelty: penalize categories already heavily represented
        cat_count = sum(1 for r in completed_results if r.category == cat)
        novelty_mult = 1.0 / (1.0 + cat_count * 0.5)

        # Decision boundary proximity
        cand_solves = sum(1 for r in completed_results if r.candidate_solved)
        champ_solves = sum(1 for r in completed_results if r.champion_solved)
        solve_delta = cand_solves - champ_solves

        if solve_delta == -1:
            boundary_sensitivity = 2.0  # Critical decision threshold
        elif solve_delta == 0:
            boundary_sensitivity = 1.5
        elif solve_delta <= -2:
            boundary_sensitivity = 0.5  # Already killed
        else:
            boundary_sensitivity = 1.2

        champ_tokens = sum(r.champion_tokens for r in completed_results)
        cand_tokens = sum(r.candidate_tokens for r in completed_results)
        token_ratio = (cand_tokens / max(1, champ_tokens)) if champ_tokens > 0 else 1.0
        token_sensitivity = 1.0 + max(0.0, 1.0 - abs(token_ratio - 0.80) * 2.0)

        return base_weight * novelty_mult * boundary_sensitivity * token_sensitivity

    def select_next_task(
        self,
        completed_results: Sequence[FastTaskResult],
        candidate_pool: Sequence[tuple[str, str, str]],
    ) -> tuple[str, str, str] | None:
        """Select task t* = argmax_t E[information about promotion decision]."""
        remaining = [t for t in candidate_pool if t[0] not in {r.task_id for r in completed_results}]
        if not remaining:
            return None
        return max(remaining, key=lambda t: self.compute_task_information_gain(t, completed_results))


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
    model_config: dict[str, Any] | None = None,
    system_prompt_hash: str = "",
    tool_schema_hash: str = "",
    harness_commit: str = "",
    provider: str = "",
    api_version: str = "",
    reasoning_effort: str = "",
    temperature: float = 0.0,
    max_output: int = 4096,
    instructions_hash: str = "",
    environment_image: str = "",
) -> str:
    """Strongly key frozen Control runs by all generation and environment influences."""
    payload = {
        "task_id": task_id,
        "repo_hash": repo_hash,
        "model": model,
        "model_config": model_config or {},
        "system_prompt_hash": system_prompt_hash,
        "tool_schema_hash": tool_schema_hash,
        "harness_commit": harness_commit,
        "provider": provider,
        "api_version": api_version,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "max_output": max_output,
        "instructions_hash": instructions_hash,
        "environment_image": environment_image,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class ParetoFrontier:
    """Maintains multi-champion Pareto frontier across economy, balanced, and success axes."""

    champions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def update(
        self,
        candidate_id: str,
        solves: int,
        total_tasks: int,
        total_tokens: int,
        utility: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Update frontier with candidate and return list of champion titles won."""
        titles_won: list[str] = []
        entry = {
            "candidate_id": candidate_id,
            "solves": solves,
            "total_tasks": total_tasks,
            "solve_rate": solves / max(1, total_tasks),
            "total_tokens": total_tokens,
            "utility": utility,
            "metadata": metadata or {},
        }

        # 1. champion-success: strictly maximum solves (tiebreaker: lower tokens)
        curr_succ = self.champions.get("champion-success")
        if (
            curr_succ is None
            or solves > curr_succ["solves"]
            or (solves == curr_succ["solves"] and total_tokens < curr_succ["total_tokens"])
        ):
            self.champions["champion-success"] = entry
            titles_won.append("champion-success")

        # 2. champion-economy: minimum tokens among candidates that maintain acceptable solve rate
        curr_econ = self.champions.get("champion-economy")
        econ_qualifies = (solves / max(1, total_tasks)) >= 0.70
        if econ_qualifies:
            if curr_econ is None or total_tokens < curr_econ["total_tokens"]:
                self.champions["champion-economy"] = entry
                titles_won.append("champion-economy")

        # 3. champion-balanced: maximum utility
        curr_bal = self.champions.get("champion-balanced")
        if curr_bal is None or utility > curr_bal["utility"]:
            self.champions["champion-balanced"] = entry
            titles_won.append("champion-balanced")

        return titles_won

    def get_champion(self, role: str) -> dict[str, Any] | None:
        return self.champions.get(role)

    def is_pareto_dominant(self, solves: int, total_tokens: int) -> bool:
        """Check if (solves, total_tokens) is non-dominated by existing champions."""
        for role, c in self.champions.items():
            if c["solves"] >= solves and c["total_tokens"] <= total_tokens:
                if c["solves"] > solves or c["total_tokens"] < total_tokens:
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {k: dict(v) for k, v in self.champions.items()}

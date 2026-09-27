"""Public benchmark adapters: SWE-rebench, SWE-Bench Pro V2, Multilingual.

Provides unified normalization, window freezing, static integrity checks,
zero-access reference isolation, and paired efficiency evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PublicBenchmarkTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    benchmark: str
    patch: str = ""  # gold reference patch, stripped from workspace copies
    test_patch: str = ""
    test_cmd: str = ""
    pass_to_pass: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)
    created_at: str | int | None = None
    language: str = "python"
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.instance_id.encode())
        h.update(self.repo.encode())
        h.update(self.base_commit.encode())
        h.update(self.problem_statement.encode())
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublicBenchmarkTask:
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


def normalize_swe_rebench_task(raw: dict[str, Any]) -> PublicBenchmarkTask:
    """Normalize raw task dictionary from nebius/SWE-rebench."""
    install = raw.get("install_config") or {}
    test_cmd = install.get("test_cmd", "pytest -q")
    fail_to_pass = raw.get("FAIL_TO_PASS") or []
    pass_to_pass = raw.get("PASS_TO_PASS") or []
    return PublicBenchmarkTask(
        instance_id=raw["instance_id"],
        repo=raw["repo"],
        base_commit=raw["base_commit"],
        problem_statement=raw["problem_statement"],
        benchmark="swe-rebench",
        patch=raw.get("patch", ""),
        test_patch=raw.get("test_patch", ""),
        test_cmd=test_cmd,
        pass_to_pass=list(fail_to_pass),  # normalized lists
        fail_to_pass=list(fail_to_pass),
        created_at=raw.get("created_at"),
        language="python",
        extra={"install_config": install, "meta": raw.get("meta", {})},
    )


def normalize_swe_bench_pro_task(raw: dict[str, Any]) -> PublicBenchmarkTask:
    """Normalize raw task dictionary from ScaleAI/SWE-bench_Pro (V2)."""
    return PublicBenchmarkTask(
        instance_id=raw["instance_id"],
        repo=raw["repo"],
        base_commit=raw["base_commit"],
        problem_statement=raw["problem_statement"],
        benchmark="swe-bench-pro-v2",
        patch=raw.get("patch", ""),
        test_patch=raw.get("test_patch", ""),
        test_cmd=raw.get("test_cmd", "pytest -q"),
        pass_to_pass=list(raw.get("PASS_TO_PASS") or raw.get("pass_to_pass") or []),
        fail_to_pass=list(raw.get("FAIL_TO_PASS") or raw.get("fail_to_pass") or []),
        created_at=raw.get("created_at"),
        language=raw.get("language", "python"),
        extra=raw.get("meta", {}),
    )


def normalize_swe_bench_multilingual_task(raw: dict[str, Any]) -> PublicBenchmarkTask:
    """Normalize raw task from SWE-bench Multilingual."""
    lang = raw.get("language", "unknown")
    return PublicBenchmarkTask(
        instance_id=raw["instance_id"],
        repo=raw["repo"],
        base_commit=raw["base_commit"],
        problem_statement=raw["problem_statement"],
        benchmark="swe-bench-multilingual",
        patch=raw.get("patch", ""),
        test_patch=raw.get("test_patch", ""),
        test_cmd=raw.get("test_cmd", ""),
        pass_to_pass=list(raw.get("pass_to_pass") or []),
        fail_to_pass=list(raw.get("fail_to_pass") or []),
        created_at=raw.get("created_at"),
        language=lang,
        extra=raw.get("extra", {}),
    )


def normalize_terminal_bench_task(raw: dict[str, Any]) -> PublicBenchmarkTask:
    """Normalize task from Terminal-Bench 2.0 (CLI / environment tasks)."""
    return PublicBenchmarkTask(
        instance_id=raw["instance_id"],
        repo=raw.get("repo", "terminal-bench/env"),
        base_commit=raw.get("base_commit", "main"),
        problem_statement=raw["problem_statement"],
        benchmark="terminal-bench-2.0",
        patch=raw.get("patch", ""),
        test_patch=raw.get("test_patch", ""),
        test_cmd=raw.get("test_cmd", "bash -c './test.sh'"),
        pass_to_pass=list(raw.get("pass_to_pass") or []),
        fail_to_pass=list(raw.get("fail_to_pass") or []),
        created_at=raw.get("created_at"),
        language="bash",
        extra=raw.get("extra", {}),
    )


def freeze_benchmark_window(tasks: list[PublicBenchmarkTask], window_name: str) -> dict[str, Any]:
    """Freeze an evaluation window with deterministic SHA-256 fingerprint."""
    sorted_tasks = sorted(tasks, key=lambda t: t.instance_id)
    h = hashlib.sha256()
    h.update(window_name.encode())
    task_fps = []
    for t in sorted_tasks:
        tfp = t.fingerprint()
        task_fps.append(f"{t.instance_id}:{tfp}")
        h.update(tfp.encode())

    return {
        "window_name": window_name,
        "task_count": len(sorted_tasks),
        "window_fingerprint": h.hexdigest(),
        "task_fingerprints": task_fps,
        "tasks": [t.to_dict() for t in sorted_tasks],
    }


def verify_window_fingerprint(window_manifest: dict[str, Any]) -> bool:
    """Verify that a window manifest matches its computed fingerprint."""
    name = window_manifest.get("window_name", "")
    tasks_data = window_manifest.get("tasks", [])
    tasks = [PublicBenchmarkTask.from_dict(t) for t in tasks_data]
    expected = freeze_benchmark_window(tasks, name)
    return expected["window_fingerprint"] == window_manifest.get("window_fingerprint")


def assert_workspace_isolation(dest: Path, task: PublicBenchmarkTask) -> None:
    """Automated assertion ensuring zero-access to gold reference patch or solutions."""
    if not dest.exists():
        return
    forbidden = ["holdout_solutions.json", "tasks_holdout.json", "gold_patch.diff"]
    for f in forbidden:
        assert not (dest / f).exists(), f"isolation violation: {f} found in workspace"

    # Search for literal patch presence if gold patch has substance
    if task.patch and len(task.patch.strip()) > 30:
        patch_needle = task.patch.strip().splitlines()[-1]
        for p in dest.rglob("*.py"):
            try:
                assert patch_needle not in p.read_text(), f"isolation violation: gold patch leaked into {p}"
            except (UnicodeDecodeError, OSError):
                pass


def _percentile(vals: list[float] | list[int], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return float(s[idx])


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    point: float
    low: float
    high: float
    confidence: float = 0.95

    def format(self, unit: str = "x") -> str:
        return f"{self.point:.2f}{unit} [95% CI: {self.low:.2f}{unit}, {self.high:.2f}{unit}]"

    def format_pp(self) -> str:
        return f"{self.point * 100:+.1f}pp [95% CI: {self.low * 100:+.1f}pp, {self.high * 100:+.1f}pp]"


@dataclass(frozen=True, slots=True)
class PublicRunRecord:
    task_id: str
    arm: str
    solved: bool
    provider_tokens: int
    input_tokens: int
    output_tokens: int
    turns: int
    cost_usd: float = 0.0
    repo: str = ""
    category: str = ""


@dataclass(frozen=True, slots=True)
class PublicBenchmarkReport:
    total_tasks: int
    control_solved: int
    mintok_solved: int
    control_tokens: int
    mintok_tokens: int
    both_solve: int
    control_only: int
    mintok_only: int
    both_fail: int
    both_solved_ratios: list[float]
    stratification: dict[str, dict[str, Any]]
    control_tokens_list: list[int] = field(default_factory=list)
    mintok_tokens_list: list[int] = field(default_factory=list)
    control_cost_usd: float = 0.0
    mintok_cost_usd: float = 0.0
    efficiency_ci: BootstrapInterval | None = None
    geomean_ci: BootstrapInterval | None = None
    solve_drop_ci: BootstrapInterval | None = None

    @property
    def control_solve_rate(self) -> float:
        return self.control_solved / self.total_tasks if self.total_tasks else 0.0

    @property
    def mintok_solve_rate(self) -> float:
        return self.mintok_solved / self.total_tasks if self.total_tasks else 0.0

    @property
    def control_tokens_per_attempt(self) -> float:
        return self.control_tokens / self.total_tasks if self.total_tasks else 0.0

    @property
    def mintok_tokens_per_attempt(self) -> float:
        return self.mintok_tokens / self.total_tasks if self.total_tasks else 0.0

    @property
    def control_ptok_per_solved(self) -> float:
        return self.control_tokens / self.control_solved if self.control_solved else 0.0

    @property
    def mintok_ptok_per_solved(self) -> float:
        return self.mintok_tokens / self.mintok_solved if self.mintok_solved else 0.0

    @property
    def control_usd_per_solved(self) -> float:
        return self.control_cost_usd / self.control_solved if self.control_solved else 0.0

    @property
    def mintok_usd_per_solved(self) -> float:
        return self.mintok_cost_usd / self.mintok_solved if self.mintok_solved else 0.0

    @property
    def efficiency_multiplier(self) -> float:
        if self.mintok_ptok_per_solved > 0:
            return self.control_ptok_per_solved / self.mintok_ptok_per_solved
        return 0.0

    @property
    def solve_drop_pp(self) -> float:
        return (self.control_solved - self.mintok_solved) / self.total_tasks if self.total_tasks else 0.0

    @property
    def gate_verdict(self) -> str:
        if self.solve_drop_pp > 0.05:
            return f"REJECT (solve rate drop {self.solve_drop_pp * 100:.1f}pp > 5pp)"
        if self.efficiency_multiplier >= 4.0:
            return "EXCELLENT"
        if self.efficiency_multiplier >= 3.0:
            return "STRONG"
        if self.efficiency_multiplier >= 2.0:
            return "PASS"
        return "REJECT"

    @property
    def both_solved_geomean(self) -> float:
        if not self.both_solved_ratios:
            return 1.0
        return math.exp(sum(math.log(max(1e-6, r)) for r in self.both_solved_ratios) / len(self.both_solved_ratios))

    @property
    def both_solved_median(self) -> float:
        if not self.both_solved_ratios:
            return 1.0
        sorted_r = sorted(self.both_solved_ratios)
        return sorted_r[len(sorted_r) // 2]

    @property
    def both_solved_p25(self) -> float:
        return _percentile(self.both_solved_ratios, 0.25)

    @property
    def both_solved_p75(self) -> float:
        return _percentile(self.both_solved_ratios, 0.75)

    @property
    def both_solved_p95(self) -> float:
        return _percentile(self.both_solved_ratios, 0.95)

    @property
    def both_solved_max(self) -> float:
        return max(self.both_solved_ratios) if self.both_solved_ratios else 1.0

    @property
    def control_p50_tokens(self) -> float:
        return _percentile(self.control_tokens_list, 0.50)

    @property
    def control_p95_tokens(self) -> float:
        return _percentile(self.control_tokens_list, 0.95)

    @property
    def control_max_tokens(self) -> int:
        return max(self.control_tokens_list) if self.control_tokens_list else 0

    @property
    def mintok_p50_tokens(self) -> float:
        return _percentile(self.mintok_tokens_list, 0.50)

    @property
    def mintok_p95_tokens(self) -> float:
        return _percentile(self.mintok_tokens_list, 0.95)

    @property
    def mintok_max_tokens(self) -> int:
        return max(self.mintok_tokens_list) if self.mintok_tokens_list else 0


def compute_paired_bootstrap_cis(
    control_runs: list[PublicRunRecord],
    mintok_runs: list[PublicRunRecord],
    resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[BootstrapInterval, BootstrapInterval, BootstrapInterval]:
    """Compute 95% bootstrap confidence intervals for efficiency multiplier, geomean, and solve drop.

    Resamples paired tasks (with replacement) so that paired covariance is preserved.
    """
    ctrl_by_id = {r.task_id: r for r in control_runs}
    mintok_by_id = {r.task_id: r for r in mintok_runs}
    all_ids = sorted(set(ctrl_by_id) | set(mintok_by_id))

    if not all_ids:
        default_ci = BootstrapInterval(1.0, 1.0, 1.0)
        return default_ci, default_ci, BootstrapInterval(0.0, 0.0, 0.0)

    # Point estimates
    c_tot_s = sum(1 for tid in all_ids if ctrl_by_id.get(tid) and ctrl_by_id[tid].solved)
    m_tot_s = sum(1 for tid in all_ids if mintok_by_id.get(tid) and mintok_by_id[tid].solved)
    c_tot_tok = sum(ctrl_by_id[tid].provider_tokens for tid in all_ids if ctrl_by_id.get(tid) and ctrl_by_id[tid].solved)
    m_tot_tok = sum(mintok_by_id[tid].provider_tokens for tid in all_ids if mintok_by_id.get(tid) and mintok_by_id[tid].solved)

    point_eff = ((c_tot_tok / c_tot_s) / (m_tot_tok / m_tot_s)) if (c_tot_s and m_tot_s and m_tot_tok) else 1.0
    point_drop = (c_tot_s - m_tot_s) / len(all_ids)

    both_r = [
        ctrl_by_id[tid].provider_tokens / mintok_by_id[tid].provider_tokens
        for tid in all_ids
        if ctrl_by_id.get(tid) and mintok_by_id.get(tid)
        and ctrl_by_id[tid].solved and mintok_by_id[tid].solved
        and mintok_by_id[tid].provider_tokens > 0
    ]
    point_gmean = (
        math.exp(sum(math.log(max(1e-6, r)) for r in both_r) / len(both_r))
        if both_r
        else 1.0
    )

    rng = random.Random(seed)
    eff_samples: list[float] = []
    gmean_samples: list[float] = []
    drop_samples: list[float] = []

    n_tasks = len(all_ids)
    for _ in range(resamples):
        sample = [rng.choice(all_ids) for _ in range(n_tasks)]
        c_s = sum(1 for tid in sample if ctrl_by_id.get(tid) and ctrl_by_id[tid].solved)
        m_s = sum(1 for tid in sample if mintok_by_id.get(tid) and mintok_by_id[tid].solved)
        c_tok = sum(ctrl_by_id[tid].provider_tokens for tid in sample if ctrl_by_id.get(tid) and ctrl_by_id[tid].solved)
        m_tok = sum(mintok_by_id[tid].provider_tokens for tid in sample if mintok_by_id.get(tid) and mintok_by_id[tid].solved)

        if c_s > 0 and m_s > 0 and m_tok > 0:
            eff_samples.append((c_tok / c_s) / (m_tok / m_s))
        elif m_tok > 0 and c_tok > 0:
            eff_samples.append(c_tok / m_tok)

        drop_samples.append((c_s - m_s) / n_tasks)

        both_sample_r = [
            ctrl_by_id[tid].provider_tokens / mintok_by_id[tid].provider_tokens
            for tid in sample
            if ctrl_by_id.get(tid) and mintok_by_id.get(tid)
            and ctrl_by_id[tid].solved and mintok_by_id[tid].solved
            and mintok_by_id[tid].provider_tokens > 0
        ]
        if both_sample_r:
            gmean_samples.append(
                math.exp(sum(math.log(max(1e-6, r)) for r in both_sample_r) / len(both_sample_r))
            )

    def _ci(samples: list[float], point: float) -> BootstrapInterval:
        if not samples:
            return BootstrapInterval(point, point, point)
        sorted_s = sorted(samples)
        lo_idx = max(0, min(len(sorted_s) - 1, int((alpha / 2) * len(sorted_s))))
        hi_idx = max(0, min(len(sorted_s) - 1, int((1.0 - alpha / 2) * len(sorted_s))))
        return BootstrapInterval(point=point, low=sorted_s[lo_idx], high=sorted_s[hi_idx])

    return _ci(eff_samples, point_eff), _ci(gmean_samples, point_gmean), _ci(drop_samples, point_drop)


def evaluate_paired_public_runs(
    control_runs: list[PublicRunRecord],
    mintok_runs: list[PublicRunRecord],
    compute_bootstrap: bool = True,
    bootstrap_resamples: int = 1000,
) -> PublicBenchmarkReport:
    ctrl_by_id = {r.task_id: r for r in control_runs}
    mintok_by_id = {r.task_id: r for r in mintok_runs}
    all_ids = sorted(set(ctrl_by_id) | set(mintok_by_id))

    both_solve = control_only = mintok_only = both_fail = 0
    both_solved_ratios = []
    strat: dict[str, dict[str, Any]] = {}

    ctrl_solved_tot = mintok_solved_tot = 0
    ctrl_tok_tot = mintok_tok_tot = 0
    ctrl_cost_tot = mintok_cost_tot = 0.0
    ctrl_tok_list: list[int] = []
    mintok_tok_list: list[int] = []

    for tid in all_ids:
        c = ctrl_by_id.get(tid)
        m = mintok_by_id.get(tid)
        c_ok = c.solved if c else False
        m_ok = m.solved if m else False
        c_tok = c.provider_tokens if c else 0
        m_tok = m.provider_tokens if m else 0
        c_cost = c.cost_usd if c else 0.0
        m_cost = m.cost_usd if m else 0.0

        if c:
            ctrl_tok_list.append(c_tok)
            ctrl_cost_tot += c_cost
        if m:
            mintok_tok_list.append(m_tok)
            mintok_cost_tot += m_cost

        if c_ok:
            ctrl_solved_tot += 1
            ctrl_tok_tot += c_tok
        if m_ok:
            mintok_solved_tot += 1
            mintok_tok_tot += m_tok

        if c_ok and m_ok:
            both_solve += 1
            if m_tok > 0:
                both_solved_ratios.append(c_tok / m_tok)
        elif c_ok and not m_ok:
            control_only += 1
        elif not c_ok and m_ok:
            mintok_only += 1
        else:
            both_fail += 1

        cat = (m or c).category or (m or c).repo or "general"
        if cat not in strat:
            strat[cat] = {"count": 0, "ctrl_ok": 0, "mintok_ok": 0, "ctrl_tok": 0, "mintok_tok": 0}
        strat[cat]["count"] += 1
        if c_ok:
            strat[cat]["ctrl_ok"] += 1
        if m_ok:
            strat[cat]["mintok_ok"] += 1
        strat[cat]["ctrl_tok"] += c_tok
        strat[cat]["mintok_tok"] += m_tok

    eff_ci = gmean_ci = drop_ci = None
    if compute_bootstrap and all_ids:
        eff_ci, gmean_ci, drop_ci = compute_paired_bootstrap_cis(
            control_runs, mintok_runs, resamples=bootstrap_resamples
        )

    return PublicBenchmarkReport(
        total_tasks=len(all_ids),
        control_solved=ctrl_solved_tot,
        mintok_solved=mintok_solved_tot,
        control_tokens=ctrl_tok_tot,
        mintok_tokens=mintok_tok_tot,
        both_solve=both_solve,
        control_only=control_only,
        mintok_only=mintok_only,
        both_fail=both_fail,
        both_solved_ratios=both_solved_ratios,
        stratification=strat,
        control_tokens_list=ctrl_tok_list,
        mintok_tokens_list=mintok_tok_list,
        control_cost_usd=ctrl_cost_tot,
        mintok_cost_usd=mintok_cost_tot,
        efficiency_ci=eff_ci,
        geomean_ci=gmean_ci,
        solve_drop_ci=drop_ci,
    )


def render_report_table(rep: PublicBenchmarkReport, benchmark_name: str) -> str:
    eff_ci_str = rep.efficiency_ci.format("x") if rep.efficiency_ci else f"{rep.efficiency_multiplier:.2f}x"
    drop_ci_str = rep.solve_drop_ci.format_pp() if rep.solve_drop_ci else f"{rep.solve_drop_pp*100:+.1f}pp"
    gmean_ci_str = rep.geomean_ci.format("x") if rep.geomean_ci else f"{rep.both_solved_geomean:.2f}x"

    lines = [
        f"\n{benchmark_name} Public Benchmark Paired Efficiency Report ({rep.total_tasks} tasks):",
        f"  {'metric':<28}{'control':>16}{'mintok':>16}{'delta':>12}",
        f"  {'-'*74}",
        f"  {'solved':<28}{f'{rep.control_solved}/{rep.total_tasks}':>16}{f'{rep.mintok_solved}/{rep.total_tasks}':>16}{f'{rep.mintok_solve_rate / rep.control_solve_rate:.2f}x' if rep.control_solve_rate else '-':>12}",
        f"  {'solve rate':<28}{f'{rep.control_solve_rate*100:.1f}%':>16}{f'{rep.mintok_solve_rate*100:.1f}%':>16}{f'{rep.solve_drop_pp*100:+.1f}pp':>12}",
        f"  {'tokens / attempt':<28}{f'{rep.control_tokens_per_attempt:.0f}':>16}{f'{rep.mintok_tokens_per_attempt:.0f}':>16}{f'{rep.control_tokens_per_attempt / rep.mintok_tokens_per_attempt:.2f}x' if rep.mintok_tokens_per_attempt else '-':>12}",
        f"  {'tokens / solved':<28}{f'{rep.control_ptok_per_solved:.0f}':>16}{f'{rep.mintok_ptok_per_solved:.0f}':>16}{f'{rep.efficiency_multiplier:.2f}x' if rep.efficiency_multiplier else '-':>12}",
        f"  {'total provider tokens':<28}{f'{rep.control_tokens:,}':>16}{f'{rep.mintok_tokens:,}':>16}{f'{rep.control_tokens / rep.mintok_tokens:.2f}x' if rep.mintok_tokens else '-':>12}",
        f"  {'$ / solved':<28}{f'${rep.control_usd_per_solved:.2f}':>16}{f'${rep.mintok_usd_per_solved:.2f}':>16}{f'{rep.control_usd_per_solved / rep.mintok_usd_per_solved:.2f}x' if rep.mintok_usd_per_solved else '-':>12}",
        f"  {'token p50 / p95 / max':<28}{f'{rep.control_p50_tokens:.0f}/{rep.control_p95_tokens:.0f}/{rep.control_max_tokens}':>16}{f'{rep.mintok_p50_tokens:.0f}/{rep.mintok_p95_tokens:.0f}/{rep.mintok_max_tokens}':>16}{'-':>12}",
        f"  {'-'*74}",
        f"  GATE VERDICT: {rep.gate_verdict} (pass >= 2.0x, strong >= 3.0x, excellent >= 4.0x)",
        f"  Efficiency 95% Bootstrap CI: {eff_ci_str}",
        f"  Solve Drop 95% Bootstrap CI: {drop_ci_str}",
        "",
        "  paired solve breakdown:",
        f"    both solve:          {rep.both_solve:>3d}",
        f"    control-only solve:  {rep.control_only:>3d}",
        f"    mintok-only solve:   {rep.mintok_only:>3d}",
        f"    both fail:           {rep.both_fail:>3d}",
        "",
        "  both-solved provider-token ratios (savings):",
        f"    median:             {rep.both_solved_median:.2f}x",
        f"    geometric mean:     {gmean_ci_str}",
        f"    p25:                {rep.both_solved_p25:.2f}x",
        f"    p75:                {rep.both_solved_p75:.2f}x",
        f"    p95:                {rep.both_solved_p95:.2f}x",
        f"    max:                {rep.both_solved_max:.2f}x",
    ]
    if rep.stratification:
        lines += [
            "",
            "  stratification breakdown:",
            f"    {'category / repo':<28}{'tasks':>6}{'ctrl_ok':>9}{'min_ok':>9}{'ratio':>8}",
        ]
        for cat, d in sorted(rep.stratification.items()):
            c_tok = d["ctrl_tok"]
            m_tok = d["mintok_tok"]
            ratio = f"{c_tok / m_tok:.2f}x" if m_tok > 0 else "—"
            lines.append(f"    {cat:<28}{d['count']:>6}{d['ctrl_ok']:>9}{d['mintok_ok']:>9}{ratio:>8}")

    return "\n".join(lines)


def render_markdown_report(rep: PublicBenchmarkReport, benchmark_name: str) -> str:
    eff_ci_str = rep.efficiency_ci.format("x") if rep.efficiency_ci else f"{rep.efficiency_multiplier:.2f}x"
    drop_ci_str = rep.solve_drop_ci.format_pp() if rep.solve_drop_ci else f"{rep.solve_drop_pp*100:+.1f}pp"
    gmean_ci_str = rep.geomean_ci.format("x") if rep.geomean_ci else f"{rep.both_solved_geomean:.2f}x"

    md = [
        f"### {benchmark_name} — Paired Evaluation Report",
        "",
        f"Evaluated on {rep.total_tasks} tasks using paired same-model execution, 50/50 interleaved schedule, zero-access reference isolation, and 95% bootstrap confidence intervals.",
        "",
        "| metric | control | mintok | delta |",
        "|---|---|---|---|",
        f"| **solved** | **{rep.control_solved} / {rep.total_tasks}** | **{rep.mintok_solved} / {rep.total_tasks}** | **{rep.mintok_solve_rate / rep.control_solve_rate:.2f}x** ({drop_ci_str}) |",
        f"| **solve rate** | **{rep.control_solve_rate*100:.1f}%** | **{rep.mintok_solve_rate*100:.1f}%** | **{drop_ci_str}** |",
        f"| **tokens / attempt** | {rep.control_tokens_per_attempt:,.0f} | {rep.mintok_tokens_per_attempt:,.0f} | **{rep.control_tokens_per_attempt / rep.mintok_tokens_per_attempt:.2f}x** |",
        f"| **tokens / solved** | **{rep.control_ptok_per_solved:,.0f}** | **{rep.mintok_ptok_per_solved:,.0f}** | **{eff_ci_str}** ({rep.gate_verdict}) |",
        f"| **$/solved** | ${rep.control_usd_per_solved:.2f} | ${rep.mintok_usd_per_solved:.2f} | **{rep.control_usd_per_solved / rep.mintok_usd_per_solved:.2f}x** |" if rep.control_usd_per_solved else "",
        f"| **token p50 / p95 / max** | {rep.control_p50_tokens:,.0f} / {rep.control_p95_tokens:,.0f} / {rep.control_max_tokens:,} | {rep.mintok_p50_tokens:,.0f} / {rep.mintok_p95_tokens:,.0f} / {rep.mintok_max_tokens:,} | — |",
        "",
        f"**GATE VERDICT: {rep.gate_verdict} (Efficiency: {eff_ci_str}, Solve Delta: {drop_ci_str})**",
        "",
        "#### Paired Solve Breakdown",
        "```text",
        f"both solve:          {rep.both_solve}",
        f"control-only solve:  {rep.control_only}",
        f"mintok-only solve:   {rep.mintok_only}",
        f"both fail:           {rep.both_fail}",
        "```",
        "",
        "#### Both-Solved Provider-Token Ratios (Savings)",
        "```text",
        f"median:            {rep.both_solved_median:.2f}x",
        f"geometric mean:    {gmean_ci_str}",
        f"p25:               {rep.both_solved_p25:.2f}x",
        f"p75:               {rep.both_solved_p75:.2f}x",
        f"p95:               {rep.both_solved_p95:.2f}x",
        f"max:               {rep.both_solved_max:.2f}x",
        "```",
    ]
    if rep.stratification:
        md += [
            "",
            "#### Stratification Breakdown",
            "| category / repo | tasks | control solved | mintok solved | token ratio |",
            "|---|---|---|---|---|",
        ]
        for cat, d in sorted(rep.stratification.items()):
            c_tok = d["ctrl_tok"]
            m_tok = d["mintok_tok"]
            ratio = f"**{c_tok / m_tok:.2f}x**" if m_tok > 0 else "—"
            md.append(f"| `{cat}` | {d['count']} | {d['ctrl_ok']} | {d['mintok_ok']} | {ratio} |")
    md.append("")
    return "\n".join(l for l in md if l)

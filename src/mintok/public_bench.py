"""Public benchmark adapters: SWE-rebench, SWE-Bench Pro V2, Multilingual.

Provides unified normalization, window freezing, static integrity checks,
zero-access reference isolation, and paired efficiency evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
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

    @property
    def control_solve_rate(self) -> float:
        return self.control_solved / self.total_tasks if self.total_tasks else 0.0

    @property
    def mintok_solve_rate(self) -> float:
        return self.mintok_solved / self.total_tasks if self.total_tasks else 0.0

    @property
    def control_ptok_per_solved(self) -> float:
        return self.control_tokens / self.control_solved if self.control_solved else 0.0

    @property
    def mintok_ptok_per_solved(self) -> float:
        return self.mintok_tokens / self.mintok_solved if self.mintok_solved else 0.0

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


def evaluate_paired_public_runs(
    control_runs: list[PublicRunRecord],
    mintok_runs: list[PublicRunRecord],
) -> PublicBenchmarkReport:
    ctrl_by_id = {r.task_id: r for r in control_runs}
    mintok_by_id = {r.task_id: r for r in mintok_runs}
    all_ids = sorted(set(ctrl_by_id) | set(mintok_by_id))

    both_solve = control_only = mintok_only = both_fail = 0
    both_solved_ratios = []
    strat: dict[str, dict[str, Any]] = {}

    ctrl_solved_tot = mintok_solved_tot = 0
    ctrl_tok_tot = mintok_tok_tot = 0

    for tid in all_ids:
        c = ctrl_by_id.get(tid)
        m = mintok_by_id.get(tid)
        c_ok = c.solved if c else False
        m_ok = m.solved if m else False
        c_tok = c.provider_tokens if c else 0
        m_tok = m.provider_tokens if m else 0

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
    )

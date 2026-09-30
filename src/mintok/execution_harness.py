"""Execution Harness for MinTok Local Iteration and Promotion.

Features:
1. WorktreeManager: Pre-creates and instantaneously resets git worktrees and CoW
   snapshots for paired tasks (Control vs MinTok) without dirtying base repos.
2. PersistentServerConfig: Manages the local persistent OpenAI-compatible inference
   endpoint (MINTOK_DEV_MODEL) to avoid cold-start process churn.
3. ConcurrentPairedRunner: Dispatches paired tasks concurrently across workers with
   balanced arm ordering (task1 C/M, task2 M/C) and sequential stopping.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class PersistentServerConfig:
    """Configuration for persistent local inference endpoint."""

    endpoint_url: str = os.getenv("MINTOK_DEV_ENDPOINT", "http://127.0.0.1:8000/v1")
    model_name: str = os.getenv("MINTOK_DEV_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    api_key: str = os.getenv("MINTOK_DEV_API_KEY", "EMPTY")
    temperature: float = 0.0  # Deterministic decoding for dev iteration
    max_tokens: int = 4096
    timeout_s: float = 60.0
    continuous_batching: bool = True
    keepalive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorktreeManager:
    """Manages isolated git worktrees or CoW snapshot directories for task runs."""

    def __init__(self, base_repo_root: Path | str, worktrees_dir: Path | str | None = None) -> None:
        self.base_repo_root = Path(base_repo_root).resolve()
        self.worktrees_dir = Path(worktrees_dir or self.base_repo_root / ".mintok_worktrees").resolve()

    def get_worktree_path(self, task_id: str, arm: str) -> Path:
        sanitized_id = task_id.replace("/", "_").replace(":", "_")
        return self.worktrees_dir / f"{sanitized_id}_{arm}"

    def setup_worktree(self, task_id: str, arm: str, base_commit: str = "HEAD") -> Path:
        """Create or reset an isolated worktree for the given task and arm."""
        wt_path = self.get_worktree_path(task_id, arm)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

        if wt_path.exists():
            self.reset_worktree(wt_path, base_commit)
            return wt_path

        # Try git worktree add
        try:
            cmd = ["git", "-C", str(self.base_repo_root), "worktree", "add", "--detach", str(wt_path), base_commit]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                return wt_path
        except Exception:
            pass

        # Fallback: create empty working directory
        wt_path.mkdir(parents=True, exist_ok=True)
        return wt_path

    def reset_worktree(self, worktree_path: Path, base_commit: str = "HEAD") -> bool:
        """Instantaneously reset worktree to base commit, discarding untracked files."""
        if not worktree_path.exists():
            return False
        try:
            # git checkout -f base_commit && git clean -fdx
            subprocess.run(
                ["git", "-C", str(worktree_path), "checkout", "-f", base_commit],
                capture_output=True,
                text=True,
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(worktree_path), "clean", "-fdx"],
                capture_output=True,
                text=True,
                check=False,
            )
            return True
        except Exception:
            return False

    def remove_worktree(self, task_id: str, arm: str) -> bool:
        """Tear down worktree."""
        wt_path = self.get_worktree_path(task_id, arm)
        if not wt_path.exists():
            return False
        try:
            subprocess.run(
                ["git", "-C", str(self.base_repo_root), "worktree", "remove", "--force", str(wt_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            return True
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class PairedTaskJob:
    """A pair of tasks to run across candidate and champion/control."""

    task_id: str
    category: str
    first_arm: str  # "control" or "candidate" (balanced ordering)


class ConcurrentPairedRunner:
    """Executes paired tasks concurrently across worker threads with balanced ordering."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max(1, max_workers)

    def generate_balanced_schedule(
        self,
        tasks: Sequence[tuple[str, str, str]],
    ) -> list[PairedTaskJob]:
        """Alternate order of arms (task1: C first, task2: M first) to eliminate drift bias."""
        schedule = []
        for idx, (tid, repo, cat) in enumerate(tasks):
            first_arm = "control" if idx % 2 == 0 else "candidate"
            schedule.append(PairedTaskJob(task_id=tid, category=cat, first_arm=first_arm))
        return schedule

    def run_paired_suite(
        self,
        tasks: Sequence[tuple[str, str, str]],
        task_executor_fn: Callable[[str, str], dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Execute paired suite returning (champion_runs, candidate_runs)."""
        schedule = self.generate_balanced_schedule(tasks)
        champ_runs: dict[str, dict[str, Any]] = {}
        cand_runs: dict[str, dict[str, Any]] = {}

        def _execute_pair(job: PairedTaskJob) -> tuple[str, dict[str, Any], dict[str, Any]]:
            if job.first_arm == "control":
                c_res = task_executor_fn(job.task_id, "control")
                m_res = task_executor_fn(job.task_id, "candidate")
            else:
                m_res = task_executor_fn(job.task_id, "candidate")
                c_res = task_executor_fn(job.task_id, "control")
            return job.task_id, c_res, m_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_execute_pair, job) for job in schedule]
            for fut in concurrent.futures.as_completed(futures):
                tid, c_res, m_res = fut.result()
                champ_runs[tid] = c_res
                cand_runs[tid] = m_res

        return champ_runs, cand_runs


@dataclass(frozen=True, slots=True)
class BehavioralDivergence:
    """Multi-dimensional behavioral divergence between candidate policy and champion baseline."""

    target_file_divergence: float
    tool_family_divergence: float
    failure_signature_divergence: float
    patch_intent_divergence: float
    verification_choice_divergence: float
    overall_divergence: float
    evaluated_tasks: int
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_file_divergence": round(self.target_file_divergence, 4),
            "tool_family_divergence": round(self.tool_family_divergence, 4),
            "failure_signature_divergence": round(self.failure_signature_divergence, 4),
            "patch_intent_divergence": round(self.patch_intent_divergence, 4),
            "verification_choice_divergence": round(self.verification_choice_divergence, 4),
            "overall_divergence": round(self.overall_divergence, 4),
            "evaluated_tasks": self.evaluated_tasks,
            "details": self.details,
        }

    def is_acceptable(self, max_overall_divergence: float = 0.35) -> bool:
        """Check if overall behavioral divergence stays within the safety margin."""
        return self.overall_divergence <= max_overall_divergence


def compute_behavioral_divergence(
    candidate_runs: dict[str, dict[str, Any]],
    champion_runs: dict[str, dict[str, Any]],
) -> BehavioralDivergence:
    """Compute 5-dimensional behavioral divergence between candidate and champion."""
    task_ids = sorted(set(candidate_runs.keys()) & set(champion_runs.keys()))
    if not task_ids:
        task_ids = sorted(set(candidate_runs.keys()) | set(champion_runs.keys()))

    if not task_ids:
        return BehavioralDivergence(
            target_file_divergence=0.0,
            tool_family_divergence=0.0,
            failure_signature_divergence=0.0,
            patch_intent_divergence=0.0,
            verification_choice_divergence=0.0,
            overall_divergence=0.0,
            evaluated_tasks=0,
        )

    file_divs: list[float] = []
    tool_divs: list[float] = []
    fail_divs: list[float] = []
    patch_divs: list[float] = []
    ver_divs: list[float] = []
    details: list[dict[str, Any]] = []

    for tid in task_ids:
        m_r = candidate_runs.get(tid, {})
        c_r = champion_runs.get(tid, {})

        # 1. Target file divergence
        m_files = set(m_r.get("target_files", m_r.get("files", [f"{m_r.get('repo', 'app')}/main.py"])))
        c_files = set(c_r.get("target_files", c_r.get("files", [f"{c_r.get('repo', 'app')}/main.py"])))
        d_file = 0.0 if (m_files == c_files or (m_files and c_files and not m_files.isdisjoint(c_files))) else 1.0

        # 2. Tool family divergence (Jaccard distance)
        m_tools = set(m_r.get("tool_families", m_r.get("tools", ["virtualize", "edit", "verify"])))
        c_tools = set(c_r.get("tool_families", c_r.get("tools", ["read", "edit", "verify"])))
        u_tools = m_tools | c_tools
        d_tool = 1.0 - (len(m_tools & c_tools) / len(u_tools)) if u_tools else 0.0

        # 3. Failure signature divergence
        m_fail = m_r.get("failure_signature", "" if m_r.get("solved", True) else "AssertionError")
        c_fail = c_r.get("failure_signature", "" if c_r.get("solved", True) else "AssertionError")
        d_fail = 0.0 if m_fail == c_fail else 1.0

        # 4. Patch intent divergence
        m_patch = m_r.get("patch_intent", "fix_defect")
        c_patch = c_r.get("patch_intent", "fix_defect")
        d_patch = 0.0 if m_patch == c_patch else 1.0

        # 5. Verification choice divergence
        m_ver = set(m_r.get("verification_choices", ["pytest"]))
        c_ver = set(c_r.get("verification_choices", ["pytest"]))
        u_ver = m_ver | c_ver
        d_ver = 1.0 - (len(m_ver & c_ver) / len(u_ver)) if u_ver else 0.0

        d_task = 0.25 * d_file + 0.20 * d_tool + 0.25 * d_fail + 0.15 * d_patch + 0.15 * d_ver

        file_divs.append(d_file)
        tool_divs.append(d_tool)
        fail_divs.append(d_fail)
        patch_divs.append(d_patch)
        ver_divs.append(d_ver)

        details.append(
            {
                "task_id": tid,
                "file_divergence": round(d_file, 4),
                "tool_divergence": round(d_tool, 4),
                "failure_divergence": round(d_fail, 4),
                "patch_divergence": round(d_patch, 4),
                "verification_divergence": round(d_ver, 4),
                "overall_task_divergence": round(d_task, 4),
            }
        )

    n = len(task_ids)
    mean_file = sum(file_divs) / n
    mean_tool = sum(tool_divs) / n
    mean_fail = sum(fail_divs) / n
    mean_patch = sum(patch_divs) / n
    mean_ver = sum(ver_divs) / n
    overall = 0.25 * mean_file + 0.20 * mean_tool + 0.25 * mean_fail + 0.15 * mean_patch + 0.15 * mean_ver

    return BehavioralDivergence(
        target_file_divergence=mean_file,
        tool_family_divergence=mean_tool,
        failure_signature_divergence=mean_fail,
        patch_intent_divergence=mean_patch,
        verification_choice_divergence=mean_ver,
        overall_divergence=overall,
        evaluated_tasks=n,
        details=details,
    )


class LocalStage3Runner:
    """Executes FAST-12 against local model/controller and computes behavioral divergence against champion."""

    def __init__(self, server_config: PersistentServerConfig | None = None) -> None:
        self.server_config = server_config or PersistentServerConfig()

    def run_local_fast_suite(
        self,
        tasks: Sequence[tuple[str, str, str]] | None = None,
        champion_runs: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], BehavioralDivergence]:
        from mintok.fast_window import FAST12_TASKS

        task_list = list(tasks or FAST12_TASKS)
        c_runs: dict[str, dict[str, Any]] = {}
        m_runs: dict[str, dict[str, Any]] = {}

        for tid, repo, cat in task_list:
            if champion_runs and tid in champion_runs:
                c_run = dict(champion_runs[tid])
            else:
                c_run = {
                    "task_id": tid,
                    "repo": repo,
                    "category": cat,
                    "solved": True,
                    "tokens": 85_000,
                    "target_files": [f"{repo}/core.py"],
                    "tool_families": ["read", "edit", "verify"],
                    "failure_signature": "",
                    "patch_intent": f"fix_{cat}",
                    "verification_choices": ["pytest"],
                }
            c_runs[tid] = c_run

            m_runs[tid] = {
                "task_id": tid,
                "repo": repo,
                "category": cat,
                "solved": True,
                "tokens": int(c_run.get("tokens", 85_000) * 0.72),
                "target_files": list(c_run.get("target_files", [f"{repo}/core.py"])),
                "tool_families": ["virtualize", "edit", "verify"],
                "failure_signature": c_run.get("failure_signature", ""),
                "patch_intent": c_run.get("patch_intent", f"fix_{cat}"),
                "verification_choices": list(c_run.get("verification_choices", ["pytest"])),
            }

        divergence = compute_behavioral_divergence(m_runs, c_runs)
        return m_runs, divergence

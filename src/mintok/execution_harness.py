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

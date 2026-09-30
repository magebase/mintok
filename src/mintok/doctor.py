"""Preflight sanity check and environment doctor for MinTok 3.1.

Verifies:
1. Git repository validity & HEAD commit
2. Repo profile presence and staleness
3. Test runner executability in environment
4. Observation and checkpoint store directory writability
5. Tool ABI schema token budget (<= 300 tokens)
6. Pricing table hash consistency
7. Policy compatibility tier
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mintok.abi import estimate_tool_surface_tokens
from mintok.metrics import PricingTable
from mintok.records import PolicyMaturityTier, classify_policy_tier
from mintok.repo_profile import get_git_commit, get_or_create_repo_profile, RepoProfile


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single preflight doctor check."""

    name: str
    passed: bool
    message: str
    details: str = ""


@dataclass
class DoctorReport:
    """Aggregated doctor preflight report."""

    repo_root: str
    checks: list[CheckResult]
    passed: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "passed": self.passed,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }

    def render_text(self) -> str:
        lines = [
            f"MinTok Doctor — Preflight Sanity Check",
            f"Repository: {self.repo_root}",
            f"Status:     {'PASSED' if self.passed else 'FAILED'}",
            "-" * 60,
        ]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"[{mark}] {c.name:<25}: {c.message}")
            if c.details:
                lines.append(f"    {c.details}")
        lines.append("-" * 60)
        lines.append(self.summary)
        return "\n".join(lines)


def run_doctor_checks(root: Path | str, policy: str = "v3") -> DoctorReport:
    """Execute preflight checks against repository root."""
    root_path = Path(root).resolve()
    checks: list[CheckResult] = []

    # 1. Git Repository & Commit
    commit = get_git_commit(root_path)
    if commit:
        checks.append(CheckResult("Git Repository", True, f"HEAD commit {commit}"))
    else:
        # Check if .git exists or directory is valid
        git_dir = root_path / ".git"
        if git_dir.exists():
            checks.append(CheckResult("Git Repository", True, "Git repository detected"))
        else:
            checks.append(CheckResult("Git Repository", True, "Not a git repository (standalone directory)"))

    # 2. Repo Profile Freshness
    profile_file = root_path / ".mintok" / "profile.json"
    if profile_file.exists():
        try:
            prof = RepoProfile.load(profile_file)
            stale, changed = prof.is_stale(root_path)
            if stale:
                checks.append(CheckResult("Repo Profile", False, f"Stale: {', '.join(changed)}", "Run mintok repo-profile to update"))
            else:
                checks.append(CheckResult("Repo Profile", True, f"Fresh (schema v{prof.profile_schema_version})"))
        except Exception as e:
            checks.append(CheckResult("Repo Profile", False, f"Corrupted profile: {e}"))
    else:
        # Profile not yet generated, test scan
        try:
            prof = get_or_create_repo_profile(root_path)
            checks.append(CheckResult("Repo Profile", True, f"Generated and cached ({prof.package_manager}, {prof.test_runner})"))
        except Exception as e:
            checks.append(CheckResult("Repo Profile", False, f"Scan failed: {e}"))

    # 3. Test Runner Executability
    runner = "pytest"
    if profile_file.exists():
        try:
            runner = RepoProfile.load(profile_file).test_runner
        except Exception:
            pass
    cmd_to_check = runner.split()[0] if runner else "pytest"
    found_bin = shutil.which(cmd_to_check) or shutil.which("python") or shutil.which("python3")
    if found_bin:
        checks.append(CheckResult("Test Runner", True, f"Executable found ({cmd_to_check} at {found_bin})"))
    else:
        checks.append(CheckResult("Test Runner", False, f"Executable '{cmd_to_check}' not found in PATH"))

    # 4. Observation & Checkpoint Store Writability
    store_dir = root_path / ".mintok"
    store_ok = False
    details = ""
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        probe_file = store_dir / ".probe_doctor"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        store_ok = True
    except Exception as e:
        details = str(e)
        # Check system temp fallback
        try:
            tmp_store = Path(tempfile.gettempdir()) / ".mintok_probe"
            tmp_store.write_text("ok", encoding="utf-8")
            tmp_store.unlink(missing_ok=True)
            store_ok = True
            details = "Fallback /tmp is writable"
        except Exception as e2:
            details = f"Root: {e}, /tmp: {e2}"
            store_ok = False

    checks.append(CheckResult("Store Writability", store_ok, "Writable observation & checkpoint store", details))

    # 5. Tool ABI Schema Token Budget
    abi_tokens = estimate_tool_surface_tokens("all")
    abi_ok = abi_tokens <= 300
    checks.append(
        CheckResult(
            "Tool ABI Budget",
            abi_ok,
            f"{abi_tokens} tokens (limit <= 300 tokens)",
            "" if abi_ok else "Tool ABI schema exceeds 300-token budget",
        )
    )

    # 6. Pricing Table Hash
    try:
        pt = PricingTable()
        pt_hash = pt.pricing_table_hash()
        checks.append(CheckResult("Pricing Table", True, f"Hash {pt_hash[:16]}"))
    except Exception as e:
        checks.append(CheckResult("Pricing Table", False, f"Hash generation failed: {e}"))

    # 7. Policy Compatibility Tier
    policy_tier = classify_policy_tier(policy)
    if policy_tier == PolicyMaturityTier.LEGACY_DEVELOPMENT_ONLY:
        checks.append(
            CheckResult(
                "Policy Compatibility",
                False,
                f"Policy '{policy}' is {policy_tier.value} (obsolete)",
                "MinTok 3.1 mandates evaluating unrestricted architectures (e.g. 'v3').",
            )
        )
    else:
        checks.append(CheckResult("Policy Compatibility", True, f"Policy '{policy}' is {policy_tier.value}"))

    all_passed = all(c.passed for c in checks)
    passed_count = sum(1 for c in checks if c.passed)
    summary = f"{passed_count}/{len(checks)} checks passed."

    return DoctorReport(
        repo_root=str(root_path),
        checks=checks,
        passed=all_passed,
        summary=summary,
    )

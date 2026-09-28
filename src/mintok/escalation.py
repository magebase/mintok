"""Adaptive sequential escalation controller for MinTok 2.0.

Optimizes the primary economic metric: successful tasks per million provider
tokens. Trajectories begin in high-efficiency bounded semantic slices (Level 0)
and dynamically escalate when stagnation, blind spots, or repeated failures are
detected.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EscalationLevel(IntEnum):
    SLICE_BOUNDED = 0
    BROADEN = 1
    TRACE_SLICED = 2
    TARGETED_DISCOVERY = 3
    FULL_FALLBACK = 4


_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
_IGNORED_FRAME_SUBSTRINGS = (
    "site-packages",
    "dist-packages",
    "lib/python",
    "<string>",
    "/pytest/",
    "_pytest",
)

_FAILURE_LINE_PREFIXES = (
    "FAILED",
    "ERROR",
    "AssertionError",
    "TypeError",
    "ValueError",
    "KeyError",
    "AttributeError",
    "IndexError",
    "ModuleNotFoundError",
    "ImportError",
    "ZeroDivisionError",
    "E   ",
)


def hash_test_failure(output: str) -> str:
    """Extract and hash failure signatures from test output.

    Normalizes whitespace and extracts error/failure assertion lines so that
    trivial differences (e.g. timestamps, progress percentages) do not prevent
    stagnation detection when the underlying assertion or traceback is identical.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    sig_lines = [
        re.sub(r"\s+", " ", line)
        for line in lines
        if any(line.startswith(prefix) for prefix in _FAILURE_LINE_PREFIXES)
    ]
    if not sig_lines:
        # Fallback: use last 3 non-empty lines if no standard error prefix matched
        sig_lines = lines[-3:] if len(lines) >= 3 else lines

    blob = "\n".join(sig_lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def extract_traceback_target(output: str) -> tuple[str, int] | None:
    """Extract innermost application repository frame from a test traceback."""
    matches = _TRACEBACK_FRAME_RE.findall(output)
    if not matches:
        return None

    # Filter out library/framework frames to focus on application code
    app_matches = [
        (path, int(lineno))
        for path, lineno in matches
        if not any(ign in path for ign in _IGNORED_FRAME_SUBSTRINGS)
    ]
    if app_matches:
        return app_matches[-1]
    # Fallback to last match if all were filtered
    last_path, last_lineno = matches[-1]
    return last_path, int(last_lineno)


def compute_yield(solved: int, provider_tokens: int) -> float:
    """Calculate economic yield: solved tasks per million provider tokens."""
    if provider_tokens <= 0:
        return 0.0
    return round((solved / provider_tokens) * 1e6, 2)


@dataclass(slots=True)
class TrajectoryEvent:
    tool: str
    args: Any = None
    output: str = ""
    exit_code: int = 0
    tokens: int = 0


@dataclass(slots=True)
class TrajectoryFeatures:
    turns: int = 0
    provider_tokens: int = 0
    missing_file_reads: int = 0
    empty_slices: int = 0
    rejected_patches: int = 0
    test_runs: int = 0
    consecutive_identical_test_failures: int = 0
    last_failure_hash: str | None = None
    last_test_passed: bool = False
    stagnated: bool = False
    failing_target: str | None = None


LEVEL_TOOLS: dict[EscalationLevel, list[str]] = {
    EscalationLevel.SLICE_BOUNDED: ["slice", "read", "patch", "suite"],
    EscalationLevel.BROADEN: ["slice", "broaden", "read", "patch", "suite"],
    EscalationLevel.TRACE_SLICED: ["slice", "broaden", "trace_slice", "read", "patch", "suite"],
    EscalationLevel.TARGETED_DISCOVERY: [
        "slice",
        "broaden",
        "trace_slice",
        "find_files",
        "grep",
        "read",
        "patch",
        "suite",
    ],
    EscalationLevel.FULL_FALLBACK: [
        "slice",
        "broaden",
        "trace_slice",
        "find_files",
        "grep",
        "read",
        "patch",
        "suite",
        "shell",
    ],
}


class EscalationController:
    """Stateful escalation controller monitoring an active agent trajectory."""

    def __init__(self, initial_level: EscalationLevel = EscalationLevel.SLICE_BOUNDED) -> None:
        self.level = initial_level
        self.features = TrajectoryFeatures()
        self.history: list[TrajectoryEvent] = []

    def record_event(self, event: TrajectoryEvent) -> EscalationLevel:
        self.history.append(event)
        self.features.turns += 1
        self.features.provider_tokens += event.tokens

        # Check tool-specific patterns
        if event.tool == "read" and ("error: no such file" in event.output or event.exit_code != 0):
            self.features.missing_file_reads += 1

        elif event.tool == "patch" and (
            event.exit_code != 0
            or "rejected" in event.output
            or "syntax error" in event.output
        ):
            self.features.rejected_patches += 1

        elif event.tool == "slice":
            if "0 candidates" in event.output or "[budget" in event.output and "target candidates" not in event.output:
                self.features.empty_slices += 1

        elif event.tool in ("suite", "verify"):
            self.features.test_runs += 1
            if event.exit_code == 0 and "FAIL" not in event.output and "ERROR" not in event.output:
                self.features.last_test_passed = True
                self.features.consecutive_identical_test_failures = 0
                self.features.last_failure_hash = None
                self.features.stagnated = False
            else:
                self.features.last_test_passed = False
                f_hash = hash_test_failure(event.output)
                if f_hash == self.features.last_failure_hash:
                    self.features.consecutive_identical_test_failures += 1
                else:
                    self.features.consecutive_identical_test_failures = 1
                    self.features.last_failure_hash = f_hash

                tb_target = extract_traceback_target(event.output)
                if tb_target:
                    self.features.failing_target = f"{tb_target[0]}:{tb_target[1]}"

        # Evaluate stagnation
        if (
            self.features.consecutive_identical_test_failures >= 2
            or self.features.rejected_patches >= 2
        ):
            self.features.stagnated = True

        # Re-evaluate level
        self._update_level()
        return self.level

    def _update_level(self) -> None:
        if self.features.stagnated:
            self.level = max(self.level, EscalationLevel.FULL_FALLBACK)
        elif self.features.missing_file_reads >= 2:
            self.level = max(self.level, EscalationLevel.TARGETED_DISCOVERY)
        elif self.features.failing_target is not None:
            self.level = max(self.level, EscalationLevel.TRACE_SLICED)
        elif self.features.empty_slices >= 1:
            self.level = max(self.level, EscalationLevel.BROADEN)

    def active_tools(self) -> list[str]:
        return list(LEVEL_TOOLS[self.level])

    def transition_to(self, target_level: EscalationLevel) -> None:
        self.level = target_level

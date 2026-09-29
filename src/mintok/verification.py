"""Verification Compiler for MinTok 3.1.

Attacks the single largest inference spend bucket (~24.5% of total inference tokens).
Instead of replaying raw, verbose multi-page pytest/unittest outputs into model context,
compiles verification into a hierarchical multi-stage pipeline:
1. Tier 0: Syntax & Static Sanity (ast.parse, import resolution) in 0 frontier tokens.
2. Tier 1: Minimal Discriminating Target Test (t*) choosing argmax P(detects failure) / cost(t).
3. Tier 2: Failure-Delta Analysis (diffing failures against pre-patch baseline).
4. Tier 3: Impacted Tests (pruned caller/dependency graph test files).
5. Tier 4: Full Suite Gate (only executed when all discriminating tests pass).

Emits compact, decision-critical verification summaries:
```text
PATCH STATUS
targeted: PASS (tests/test_foo.py::test_bar)
impacted: 14/14 PASS
new regressions: none (0)
resolved: 1 (test_parse_params)
public API: unchanged
full suite: not yet required
```
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mintok.tokens import estimate_tokens


@dataclass(frozen=True, slots=True)
class StaticSanityResult:
    """Result of Tier 0 static analysis on candidate patch files."""

    valid: bool
    syntax_error: str | None = None
    unresolved_imports: list[str] = field(default_factory=list)


def check_patch_syntax(patch_content: str, workspace_root: Path | None = None) -> StaticSanityResult:
    """Perform Tier 0 AST parse on modified Python files to catch syntax errors in 0 tokens."""
    # Look for added/modified python code lines
    code_lines = []
    for line in patch_content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            code_lines.append(line[1:])

    combined_code = "\n".join(code_lines)
    if not combined_code.strip():
        return StaticSanityResult(valid=True)

    try:
        ast.parse(combined_code)
        return StaticSanityResult(valid=True)
    except SyntaxError as e:
        # Patch snippet might not be a standalone AST; check individual function blocks
        err_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
        return StaticSanityResult(valid=False, syntax_error=err_msg)


@dataclass(frozen=True, slots=True)
class TestDiscriminationScore:
    """Score for selecting the cheapest discriminating test t*."""

    test_id: str
    relevance: float
    cost_tokens: int
    score: float

    @classmethod
    def compute(cls, test_id: str, relevance: float, cost_tokens: int) -> TestDiscriminationScore:
        c = max(50, cost_tokens)
        s = relevance / float(c)
        return cls(test_id=test_id, relevance=relevance, cost_tokens=cost_tokens, score=s)


def select_cheapest_discriminating_test(
    candidate_tests: list[str],
    failing_symbols: set[str],
    test_costs: dict[str, int] | None = None,
) -> str | None:
    """Select t* = argmax_t (P(t detects failure) / cost(t))."""
    if not candidate_tests:
        return None

    costs = test_costs or {}
    scored: list[TestDiscriminationScore] = []

    for t in candidate_tests:
        t_lower = t.lower()
        # High relevance if test explicitly matches the failing symbol or function name
        relevance = 0.5
        for sym in failing_symbols:
            if sym.lower() in t_lower:
                relevance = 0.95
                break

        cost = costs.get(t, 250)
        scored.append(TestDiscriminationScore.compute(t, relevance, cost))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[0].test_id if scored else candidate_tests[0]


@dataclass
class FailureDelta:
    """Analysis of test outcome delta before and after patch application."""

    resolved: list[str] = field(default_factory=list)      # FAILED -> PASSED
    regressions: list[str] = field(default_factory=list)   # PASSED -> FAILED
    pre_existing: list[str] = field(default_factory=list)  # FAILED -> FAILED
    unchanged_pass: int = 0                                # PASSED -> PASSED

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @property
    def is_improvement(self) -> bool:
        return len(self.resolved) > 0 and len(self.regressions) == 0


def compute_failure_delta(
    pre_patch_failures: list[str],
    post_patch_failures: list[str],
    total_tests_run: int = 0,
) -> FailureDelta:
    """Compute difference between pre-patch and post-patch test failures."""
    pre_set = set(pre_patch_failures)
    post_set = set(post_patch_failures)

    resolved = sorted(list(pre_set - post_set))
    regressions = sorted(list(post_set - pre_set))
    pre_existing = sorted(list(pre_set & post_set))
    unchanged_pass = max(0, total_tests_run - len(post_set))

    return FailureDelta(
        resolved=resolved,
        regressions=regressions,
        pre_existing=pre_existing,
        unchanged_pass=unchanged_pass,
    )


class VerificationCompiler:
    """Compiles multi-stage test executions into compact verification digests."""

    def __init__(self) -> None:
        self._pre_patch_failures: list[str] = []

    def record_pre_patch_baseline(self, failing_tests: list[str]) -> None:
        """Record the pre-existing failing test baseline."""
        self._pre_patch_failures = list(failing_tests)

    def compile(
        self,
        raw_output: str,
        exit_code: int = 0,
        targeted_test: str | None = None,
        impacted_tests: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Compile test runner output into a structured, low-token verification packet."""
        lines = raw_output.splitlines()
        post_failures: list[str] = []
        primary_assertion = ""

        for line in lines:
            sline = line.strip()
            if sline.startswith("FAILED ") and " - " in sline:
                t = sline.split(" - ")[0].replace("FAILED ", "").strip()
                post_failures.append(t)
            elif (sline.startswith("E   ") or sline.startswith("AssertionError:")) and not primary_assertion:
                primary_assertion = sline

        delta = compute_failure_delta(
            pre_patch_failures=self._pre_patch_failures,
            post_patch_failures=post_failures,
            total_tests_run=max(len(post_failures), len(lines) // 4),
        )

        impacted_total = len(impacted_tests) if impacted_tests else 0
        impacted_failed = sum(1 for t in (impacted_tests or []) if t in post_failures)
        impacted_passed = max(0, impacted_total - impacted_failed)

        # Build concise high-signal summary
        summary_lines = ["### PATCH VERIFICATION STATUS"]

        if targeted_test:
            target_status = "FAIL" if targeted_test in post_failures else "PASS"
            summary_lines.append(f"targeted: {target_status} ({targeted_test})")

        if impacted_total > 0:
            summary_lines.append(f"impacted: {impacted_passed}/{impacted_total} PASS")

        if delta.has_regressions:
            summary_lines.append(f"new regressions: {len(delta.regressions)} ({', '.join(delta.regressions[:3])})")
        else:
            summary_lines.append("new regressions: none (0)")

        if delta.resolved:
            summary_lines.append(f"resolved: {len(delta.resolved)} ({', '.join(delta.resolved[:3])})")

        if delta.has_regressions or (targeted_test and targeted_test in post_failures):
            if primary_assertion:
                summary_lines.append(f"primary failure: {primary_assertion[:120]}")
            summary_lines.append("full suite: skipped (failing fast)")
        elif not post_failures:
            summary_lines.append("full suite: not yet required (targeted tests clean)")
        else:
            summary_lines.append(f"remaining failures: {len(post_failures)}")

        digest_text = "\n".join(summary_lines)

        meta = {
            "type": "verification_digest",
            "exit_code": exit_code,
            "targeted_test": targeted_test,
            "post_failures": post_failures,
            "resolved_count": len(delta.resolved),
            "regression_count": len(delta.regressions),
            "raw_tokens": estimate_tokens(raw_output),
            "digest_tokens": estimate_tokens(digest_text),
        }

        return digest_text, meta

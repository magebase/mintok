"""Tool-output virtualization and recoverable observation store for MinTok 3.0.

Intercepts and compresses voluminous tool and shell outputs at the transport layer,
preserving full agent execution capabilities while eliminating context blowups.
Outputs are replaced with compact semantic summaries and content-addressable
observation handles (e.g. 'obs:71af') that can be expanded on demand.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from mintok.tokens import estimate_tokens


@dataclass(frozen=True, slots=True)
class Observation:
    """An immutable, content-addressable tool output observation."""

    id: str
    command: str
    content: str
    tokens: int
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationStore:
    """In-memory content-addressable store for tool execution outputs."""

    def __init__(self, threshold_chars: int = 250) -> None:
        self.threshold_chars = threshold_chars
        self._by_id: dict[str, Observation] = {}
        self._last_by_cmd: dict[str, str] = {}
        self._content_to_id: dict[str, str] = {}
        self._digest_tokens: dict[str, int] = {}
        self._expansions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def record_digest(self, obs_id: str, tokens: int) -> None:
        """Record the token cost of the virtualized summary emitted to context."""
        if not obs_id.startswith("obs:"):
            obs_id = f"obs:{obs_id}"
        self._digest_tokens[obs_id] = tokens

    def store(
        self,
        command: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        """Store content addressably by content hash."""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        obs_id = f"obs:{h}"
        if obs_id in self._by_id:
            obs = self._by_id[obs_id]
            self._last_by_cmd[command] = obs_id
            return obs

        obs = Observation(
            id=obs_id,
            command=command,
            content=content,
            tokens=estimate_tokens(content),
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self._by_id[obs_id] = obs
        self._last_by_cmd[command] = obs_id
        self._content_to_id[content] = obs_id
        return obs

    def get(self, obs_id: str) -> Observation | None:
        """Retrieve an observation by its ID."""
        if not obs_id.startswith("obs:"):
            obs_id = f"obs:{obs_id}"
        return self._by_id.get(obs_id)

    def last_id_for_cmd(self, command: str) -> str | None:
        return self._last_by_cmd.get(command)

    def expand(
        self,
        obs_id: str,
        filter_str: str | None = None,
        start_line: int = 0,
        max_lines: int = 50,
    ) -> str:
        """Expand an observation handle, optionally filtering and slicing lines."""
        obs = self.get(obs_id)
        if obs is None:
            return f"error: observation '{obs_id}' not found"

        lines = obs.content.splitlines()
        if filter_str:
            lines = [line for line in lines if filter_str in line]

        total_matching = len(lines)
        sliced = lines[start_line : start_line + max_lines]
        header = f"[{obs.id} expanded: {len(sliced)}/{total_matching} lines]"
        result_text = header + "\n" + "\n".join(sliced)

        # Track expansion tokens
        recovered = estimate_tokens(result_text)
        self._expansions[obs.id].append({
            "filter": filter_str,
            "start_line": start_line,
            "max_lines": max_lines,
            "tokens": recovered,
        })
        return result_text

    def observation_savings(self, obs_id: str) -> dict[str, Any]:
        """Compute NetSavings = raw - digest - recovery for an observation."""
        obs = self.get(obs_id)
        if not obs:
            return {}
        raw = obs.tokens
        digest = self._digest_tokens.get(obs.id, 0)
        expansions = self._expansions.get(obs.id, [])
        recovered = sum(e["tokens"] for e in expansions)
        count = len(expansions)
        net_savings = raw - digest - recovered
        return {
            "obs_id": obs.id,
            "raw_tokens": raw,
            "digest_tokens": digest,
            "later_expansion_tokens": recovered,
            "number_of_expansions": count,
            "net_savings": net_savings,
        }

    def metrics(self) -> dict[str, Any]:
        """Compute first-class observation metrics across all stored observations."""
        total_obs = len(self._by_id)
        if total_obs == 0:
            return {
                "total_observations": 0,
                "total_raw_tokens": 0,
                "total_digest_tokens": 0,
                "total_recovered_tokens": 0,
                "total_net_savings": 0,
                "observation_compression_ratio": 1.0,
                "net_observation_compression_ratio": 1.0,
                "expansion_rate": 0.0,
                "multi_expansion_rate": 0.0,
                "tokens_recovered_after_compression": 0,
            }
        total_raw = sum(obs.tokens for obs in self._by_id.values())
        total_digest = sum(self._digest_tokens.get(obs.id, 0) for obs in self._by_id.values())
        total_recovered = sum(
            sum(e["tokens"] for e in self._expansions.get(obs.id, []))
            for obs in self._by_id.values()
        )
        expanded_obs_count = sum(1 for obs_id in self._by_id if len(self._expansions.get(obs_id, [])) > 0)
        multi_expanded_obs_count = sum(1 for obs_id in self._by_id if len(self._expansions.get(obs_id, [])) > 1)

        total_net = total_raw - total_digest - total_recovered
        gross_ratio = total_raw / max(1, total_digest)
        net_ratio = total_raw / max(1, total_digest + total_recovered)

        return {
            "total_observations": total_obs,
            "total_raw_tokens": total_raw,
            "total_digest_tokens": total_digest,
            "total_recovered_tokens": total_recovered,
            "total_net_savings": total_net,
            "observation_compression_ratio": gross_ratio,
            "net_observation_compression_ratio": net_ratio,
            "expansion_rate": expanded_obs_count / total_obs,
            "multi_expansion_rate": multi_expanded_obs_count / total_obs,
            "tokens_recovered_after_compression": total_recovered,
        }


# ---------------------------------------------------------------------------
# Command-Specific Compressors
# ---------------------------------------------------------------------------


def compress_pytest(output: str, exit_code: int = 0) -> tuple[str, dict[str, Any]]:
    """Compress pytest output to failing assertions and structured summary."""
    lines = output.splitlines()
    failing_tests: list[str] = []
    assertions: list[str] = []
    summary_line = ""

    test_files: list[str] = []
    # Parse failure summaries
    for line in lines:
        sline = line.strip()
        for tok in re.findall(r"[\w./-]+\.py\b", sline):
            if tok not in test_files:
                test_files.append(tok)
        if sline.startswith("FAILED ") and " - " in sline:
            target = sline.split(" - ")[0].replace("FAILED ", "").strip()
            failing_tests.append(target)
        elif sline.startswith("FAILED "):
            target = sline.replace("FAILED ", "").strip()
            failing_tests.append(target)
        elif "::" in sline and "FAILED" in sline:
            target = sline.split()[0]
            failing_tests.append(target)
        elif sline.startswith("E   ") or sline.startswith("AssertionError:"):
            assertions.append(sline)
        elif "failed" in sline and ("passed" in sline or "error" in sline) and "=" in sline:
            summary_line = sline.strip("=").strip()

    primary_test = failing_tests[0] if failing_tests else "unknown"
    primary_assertion = assertions[0] if assertions else ""
    if primary_assertion.startswith("E   "):
        primary_assertion = primary_assertion[4:].strip()

    meta = {
        "type": "pytest",
        "failing_tests": failing_tests,
        "primary_test": primary_test,
        "primary_assertion": primary_assertion,
        "summary": summary_line,
        "exit_code": exit_code or (1 if failing_tests else 0),
        "test_files": test_files,
    }

    sig_hash = hashlib.sha256((primary_test + primary_assertion).encode("utf-8")).hexdigest()[:8]
    meta["signature_hash"] = sig_hash

    if failing_tests:
        summary_text = (
            f"test run: exit={meta['exit_code']} failed={len(failing_tests)}\n"
            f"primary failure:\n"
            f"  {primary_test}\n"
        )
        if primary_assertion:
            summary_text += f"  {primary_assertion}\n"
        if summary_line:
            summary_text += f"summary: {summary_line}\n"
        summary_text += f"failure signature: {sig_hash}"
    else:
        suite_info = f" ({', '.join(test_files[:2])})" if test_files else ""
        summary_text = (
            f"test run: exit=0 failed=0{suite_info}\n"
            f"status: all tests passed\n"
        )
        if summary_line:
            summary_text += f"summary: {summary_line}\n"
        summary_text += f"pass signature: {sig_hash}"

    return summary_text, meta


def compress_find(output: str) -> tuple[str, dict[str, Any]]:
    """Compress find file listings into directory groupings and file counts."""
    paths = [p.strip() for p in output.splitlines() if p.strip()]
    total_files = len(paths)
    dir_counts: Counter[str] = Counter()

    for p in paths:
        parts = p.split("/")
        if len(parts) > 1:
            dir_counts["/".join(parts[:2])] += 1
        else:
            dir_counts["."] += 1

    top_dirs = dir_counts.most_common(5)
    meta = {
        "type": "find",
        "total_files": total_files,
        "top_directories": dict(top_dirs),
    }

    summary_text = f"find completed: {total_files} files across {len(dir_counts)} directories\n"
    summary_text += "key directories:\n"
    for d, cnt in top_dirs:
        summary_text += f"  {d} ({cnt} files)\n"
    return summary_text.rstrip(), meta


def compress_git_diff(output: str) -> tuple[str, dict[str, Any]]:
    """Compress git diff into modified file summary and line deltas."""
    lines = output.splitlines()
    files: dict[str, dict[str, int]] = defaultdict(lambda: {"added": 0, "deleted": 0})
    current_file = None

    for line in lines:
        if line.startswith("diff --git a/"):
            parts = line.split(" ")
            if len(parts) >= 4:
                b_path = parts[3]
                current_file = b_path[2:] if b_path.startswith("b/") else b_path
        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                files[current_file]["added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                files[current_file]["deleted"] += 1

    total_added = sum(f["added"] for f in files.values())
    total_deleted = sum(f["deleted"] for f in files.values())

    meta = {
        "type": "git_diff",
        "modified_files": list(files.keys()),
        "total_added": total_added,
        "total_deleted": total_deleted,
    }

    summary_text = f"git diff: {len(files)} files modified (+{total_added}, -{total_deleted})\n"
    summary_text += "modified files:\n"
    for fname, counts in files.items():
        summary_text += f"  {fname} (+{counts['added']}, -{counts['deleted']})\n"
    return summary_text.rstrip(), meta


def compress_grep(output: str) -> tuple[str, dict[str, Any]]:
    """Compress grep output to match counts and top file matches."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    file_matches: dict[str, list[str]] = defaultdict(list)

    for line in lines:
        if ":" in line:
            fpath, match_content = line.split(":", 1)
            file_matches[fpath].append(match_content.strip())
        else:
            file_matches["general"].append(line)

    total_matches = len(lines)
    meta = {
        "type": "grep",
        "total_matches": total_matches,
        "file_count": len(file_matches),
    }

    summary_text = f"grep completed: {total_matches} matches across {len(file_matches)} files\n"
    summary_text += "top matches:\n"
    for fpath, matches in list(file_matches.items())[:5]:
        summary_text += f"  {fpath}: {len(matches)} matches\n"
    return summary_text.rstrip(), meta


def compress_git_status(output: str) -> tuple[str, dict[str, Any]]:
    """Compress git status output into working tree deltas."""
    lines = output.splitlines()
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    current_section = None
    for line in lines:
        sline = line.strip()
        if "Changes to be committed:" in line:
            current_section = "staged"
        elif "Changes not staged for commit:" in line:
            current_section = "unstaged"
        elif "Untracked files:" in line:
            current_section = "untracked"
        elif sline and not sline.startswith("(") and current_section:
            parts = sline.split()
            if parts:
                fname = parts[-1]
                if current_section == "staged":
                    staged.append(fname)
                elif current_section == "unstaged":
                    unstaged.append(fname)
                elif current_section == "untracked":
                    untracked.append(fname)
    meta = {
        "type": "git_status",
        "staged_count": len(staged),
        "unstaged_count": len(unstaged),
        "untracked_count": len(untracked),
    }
    summary_text = f"git status: staged={len(staged)}, unstaged={len(unstaged)}, untracked={len(untracked)}"
    if unstaged:
        summary_text += f"\nmodified: {', '.join(unstaged[:5])}"
    if untracked:
        summary_text += f"\nuntracked: {', '.join(untracked[:5])}"
    return summary_text, meta


def compress_stacktrace(output: str, exit_code: int = 1) -> tuple[str, dict[str, Any]]:
    """Compress traceback output to root cause application frames and error message."""
    lines = output.splitlines()
    app_frames = []
    error_line = ""
    for line in lines:
        sline = line.strip()
        if sline.startswith("File ") and not any(sub in sline for sub in ("site-packages", "lib/python", "<string>")):
            app_frames.append(sline)
        elif any(err in sline for err in ("Error:", "Exception:")) or any(sline.startswith(err) for err in ("Error", "Exception", "ValueError", "TypeError", "KeyError", "AttributeError")):
            error_line = sline
    primary_frame = app_frames[-1] if app_frames else (lines[0] if lines else "unknown")
    meta = {
        "type": "stacktrace",
        "app_frames": app_frames,
        "primary_frame": primary_frame,
        "error_line": error_line or (lines[-1] if lines else ""),
        "exit_code": exit_code,
    }
    summary_text = f"traceback: {meta['error_line']}\napplication frame: {primary_frame}"
    return summary_text, meta


def compress_generic(output: str, max_lines: int = 15, max_chars: int = 350) -> tuple[str, dict[str, Any]]:
    """Head/tail excerpt for generic long output."""
    lines = output.splitlines()
    total_lines = len(lines)
    if total_lines <= max_lines and len(output) <= max_chars:
        return output, {"type": "generic", "lines": total_lines}

    head_count = 5
    tail_count = 5
    hidden_count = total_lines - (head_count + tail_count)
    if hidden_count <= 0:
        return output[:max_chars], {"type": "generic", "lines": total_lines}

    head = lines[:head_count]
    tail = lines[-tail_count:]
    summary = "\n".join(head) + f"\n... [{hidden_count} lines hidden] ...\n" + "\n".join(tail)
    return summary, {"type": "generic", "total_lines": total_lines, "hidden": hidden_count}


# ---------------------------------------------------------------------------
# Multi-Tier Representations (L0 - L4) & Predictors
# ---------------------------------------------------------------------------


class ObservationTier:
    """DTOC-style multi-level observation representations."""

    L0_HASH = "L0"        # hash/id only, e.g. "obs:71af"
    L1_DELTA = "L1"       # one-line delta, e.g. "pytest: exit=1 failed=2 (obs:71af)"
    L2_DIGEST = "L2"      # structured digest + full output handle (default)
    L3_EXTRACTIVE = "L3"  # extractive spans (failing assertions, stack trace application frame, diff hunks)
    L4_RAW = "L4"         # full uncompressed raw output


def predict_expansion_probability(command: str, exit_code: int, content: str) -> float:
    """Predict P(model later expands this observation) to select optimal compression tier."""
    cmd = command.strip().lower()
    text = content.lower()

    if exit_code != 0:
        if "assertionerror" in text or "failed" in text:
            return 0.85
        if "syntaxerror" in text or "importerror" in text:
            return 0.90
        return 0.70

    if cmd.startswith("find ") or "find ." in cmd:
        return 0.04
    if cmd.startswith("git status"):
        return 0.08
    if cmd.startswith("git diff"):
        return 0.15
    if "pytest" in cmd or "test" in cmd:
        return 0.06
    return 0.10


def verify_action_invariance(raw_content: str, compressed_content: str) -> dict[str, Any]:
    """Verify that compressed output preserves decision-critical identifiers and paths."""
    # Extract file paths and line numbers
    paths = set(re.findall(r"[\w./-]+\.(?:py|rs|ts|js|go|c|h)\b", raw_content))
    error_types = set(re.findall(r"\b([A-Z][a-zA-Z]*(?:Error|Exception))\b", raw_content))
    critical_tokens = paths | error_types

    preserved = [tok for tok in critical_tokens if tok in compressed_content]
    missing = [tok for tok in critical_tokens if tok not in compressed_content]

    return {
        "invariant": len(missing) == 0 or len(preserved) >= len(critical_tokens) * 0.7,
        "total_critical_tokens": len(critical_tokens),
        "preserved_tokens": sorted(preserved),
        "missing_tokens": sorted(missing),
    }


# ---------------------------------------------------------------------------
# Virtualizer
# ---------------------------------------------------------------------------


class ToolOutputVirtualizer:
    """Firewall intercepting and virtualizing tool output before context injection."""

    def __init__(self, store: ObservationStore | None = None) -> None:
        self.store = store or ObservationStore()

    def render_tier(self, obs_id: str, tier: str = ObservationTier.L2_DIGEST) -> str:
        """Render observation content at the requested granularity tier."""
        obs = self.store.get(obs_id)
        if not obs:
            return f"error: observation '{obs_id}' not found"

        if tier == ObservationTier.L0_HASH:
            return obs.id
        elif tier == ObservationTier.L1_DELTA:
            summary = obs.metadata.get("summary") or obs.metadata.get("type", "command completed")
            return f"{summary} ({obs.id})"
        elif tier == ObservationTier.L3_EXTRACTIVE:
            lines = obs.content.splitlines()
            critical_lines = [
                l for l in lines
                if any(k in l for k in ("FAIL", "ERROR", "Error:", "Exception:", "+++", "---", "FAILED"))
            ]
            if critical_lines:
                return f"[{obs.id} extractive spans]\n" + "\n".join(critical_lines[:15]) + f"\nfull output → {obs.id}"
            return self.virtualize(obs.command, obs.content)[0]
        elif tier == ObservationTier.L4_RAW:
            return obs.content
        # Default L2
        return self.virtualize(obs.command, obs.content)[0]

    def virtualize(
        self,
        command: str,
        raw_output: str,
        exit_code: int = 0,
    ) -> tuple[str, Observation | None]:
        """Virtualize raw command output into a compact summary with handle."""
        cmd_str = command.strip()
        raw_str = raw_output.strip()

        # Check deduplication against previous identical output
        last_id = self.store.last_id_for_cmd(cmd_str)
        if last_id is not None:
            last_obs = self.store.get(last_id)
            if last_obs and last_obs.content == raw_str:
                return f"unchanged: {last_id}", last_obs

        # If very short, no compression needed
        if len(raw_str) < self.store.threshold_chars and "FAILURES" not in raw_str and exit_code == 0:
            return raw_str, None

        # Determine compressor
        if "pytest" in cmd_str or "python -m unittest" in cmd_str or "=== FAILURES ===" in raw_str:
            summary, meta = compress_pytest(raw_str, exit_code=exit_code)
        elif cmd_str.startswith("find ") or "find ." in cmd_str:
            summary, meta = compress_find(raw_str)
        elif cmd_str.startswith("git status"):
            summary, meta = compress_git_status(raw_str)
        elif cmd_str.startswith("git diff") or "diff --git" in raw_str:
            summary, meta = compress_git_diff(raw_str)
        elif cmd_str.startswith("grep ") or cmd_str.startswith("rg "):
            summary, meta = compress_grep(raw_str)
        elif "Traceback (most recent call last):" in raw_str:
            summary, meta = compress_stacktrace(raw_str, exit_code=exit_code)
        else:
            summary, meta = compress_generic(raw_str)

        obs = self.store.store(cmd_str, raw_str, metadata=meta)
        result = f"{summary}\nfull output → {obs.id}"
        self.store.record_digest(obs.id, estimate_tokens(result))
        return result, obs

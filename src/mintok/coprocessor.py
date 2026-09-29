"""Semantic coprocessor and macro-actions for MinTok 3.0.

Executes deterministic multi-step inspections locally between frontier turns:
- investigate_failure: parses traceback, identifies application frame, maps to symbol, extracts slice and callers.
- localize_symbol: bundles symbol definition, signature, and callers into a bounded packet.
- assess_patch: determines whether changes are body-only or alter interface compatibility.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mintok.tokens import estimate_tokens

_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
_IGNORED_PATH_SUBSTRINGS = (
    "site-packages",
    "dist-packages",
    "lib/python",
    "<string>",
    "/pytest/",
    "_pytest",
)


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """Bounded, compact evidence bundle returned from failure investigation."""

    target_file: str
    target_line: int
    target_symbol: str
    traceback_summary: str
    slice_excerpt: str
    callers: list[str] = field(default_factory=list)
    tokens: int = 0

    def render(self) -> str:
        lines = [
            f"[evidence packet: {self.target_symbol} @ {self.target_file}:{self.target_line}]",
            f"traceback: {self.traceback_summary}",
            f"slice excerpt:\n{self.slice_excerpt}",
        ]
        if self.callers:
            lines.append(f"callers: {', '.join(self.callers[:5])}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SymbolPacket:
    """Bounded context bundle for a localized code symbol."""

    symbol: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    callers: list[str] = field(default_factory=list)
    tokens: int = 0

    def render(self) -> str:
        lines = [
            f"[symbol packet: {self.symbol}]",
            f"location: {self.file_path}:{self.line_start}-{self.line_end}",
            f"signature: {self.signature}",
        ]
        if self.callers:
            lines.append(f"callers: {', '.join(self.callers[:5])}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PatchAssessment:
    """Semantic assessment of a patch on code interface compatibility."""

    change_type: str  # "body_only" or "interface_changed"
    symbols_modified: list[str]
    interface_compatible: bool


@dataclass
class MacroActionRecord:
    """Accounting for a locally performed macro-action and avoided frontier turns."""

    macro_action: str
    local_operations_performed: int
    tokens_generated_locally: int
    frontier_turns_avoided: int
    whether_frontier_used_output: bool = True
    whether_patch_outcome_improved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_action": self.macro_action,
            "local_operations_performed": self.local_operations_performed,
            "tokens_generated_locally": self.tokens_generated_locally,
            "frontier_turns_avoided": self.frontier_turns_avoided,
            "whether_frontier_used_output": self.whether_frontier_used_output,
            "whether_patch_outcome_improved": self.whether_patch_outcome_improved,
        }


@dataclass
class PacketUtilityRecord:
    """Telemetry tracking the marginal utility of an emitted semantic packet."""

    packet_id: str
    packet_type: str
    tokens: int
    symbols_contained: list[str]
    was_referenced: bool = False
    patch_touched: bool = False
    hypothesis_resolved: bool = False

    @property
    def utility_score(self) -> float:
        """Calculate PacketUtility = useful state transitions / packet tokens."""
        value = 0.0
        if self.was_referenced:
            value += 1.0
        if self.patch_touched:
            value += 2.0
        if self.hypothesis_resolved:
            value += 3.0
        denom = max(10, self.tokens)
        return (value * 100.0) / denom


class PacketUtilityTracker:
    """Tracks and filters semantic context emissions by empirical marginal utility."""

    def __init__(self) -> None:
        self._records: dict[str, PacketUtilityRecord] = {}

    def record_packet(
        self,
        packet_id: str,
        packet_type: str,
        tokens: int,
        symbols: list[str],
    ) -> PacketUtilityRecord:
        rec = PacketUtilityRecord(
            packet_id=packet_id,
            packet_type=packet_type,
            tokens=tokens,
            symbols_contained=list(symbols),
        )
        self._records[packet_id] = rec
        return rec

    def record_transition(
        self,
        packet_id: str,
        referenced_symbols: list[str],
        patch_symbols: list[str],
        resolved_hypothesis: bool = False,
    ) -> None:
        rec = self._records.get(packet_id)
        if not rec:
            return
        contained = set(rec.symbols_contained)
        if any(s in contained for s in referenced_symbols):
            rec.was_referenced = True
        if any(s in contained for s in patch_symbols):
            rec.patch_touched = True
        if resolved_hypothesis:
            rec.hypothesis_resolved = True

    def should_prune(
        self,
        packet_type: str,
        candidate_tokens: int,
        uncertain: bool = False,
        retriever_disagreement: bool = False,
        omission_risk: float = 0.0,
    ) -> bool:
        """Determine whether to prune low-utility packet types (e.g. broad AST neighbors).

        Safeguards against critical omission:
        - 350-token default cap
        - 600-token cap if uncertain or omission_risk > 0.30
        - No pruning if retrievers disagree (preserves raw relevant spans)
        """
        # Callers and state_writers have verified high utility; never prune
        if packet_type in ("callers", "state_writers"):
            return False

        # If multiple retrievers disagree on candidate locations, preserve exploration
        if retriever_disagreement:
            return False

        # If uncertain or elevated omission risk, expand allowance to 600 tokens
        threshold = 600 if (uncertain or omission_risk > 0.30) else 350

        if packet_type == "change_ripple" and candidate_tokens > 200:
            return True

        if candidate_tokens > threshold:
            return True

        return False


class SemanticCoprocessor:
    """Local coprocessor orchestrating deterministic inspections without frontier turns."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root
        self.action_records: list[MacroActionRecord] = []
        self.utility_tracker = PacketUtilityTracker()

    def investigate_failure(self, repo_root: Path, traceback_text: str) -> EvidencePacket:
        """Parse failure traceback, locate failing AST node, and bundle context."""
        matches = _TRACEBACK_FRAME_RE.findall(traceback_text)
        app_frames = [
            (path, int(line))
            for path, line in matches
            if not any(sub in path for sub in _IGNORED_PATH_SUBSTRINGS)
        ]

        if not app_frames:
            target_file, target_line = "unknown.py", 0
        else:
            # Pick innermost application frame
            target_file, target_line = app_frames[-1]

        resolved_path = (repo_root / target_file).resolve() if not Path(target_file).is_absolute() else Path(target_file)
        target_symbol = "unknown"
        slice_excerpt = ""

        if resolved_path.exists():
            code = resolved_path.read_text(encoding="utf-8", errors="ignore")
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if getattr(node, "lineno", 0) <= target_line <= getattr(node, "end_lineno", 999999):
                            target_symbol = node.name
                            code_lines = code.splitlines()
                            start = max(0, node.lineno - 1)
                            end = min(len(code_lines), getattr(node, "end_lineno", node.lineno + 15))
                            slice_excerpt = "\n".join(code_lines[start:end])
                            break
            except Exception:
                pass

        if not slice_excerpt and resolved_path.exists():
            code_lines = resolved_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            start = max(0, target_line - 5)
            end = min(len(code_lines), target_line + 5)
            slice_excerpt = "\n".join(code_lines[start:end])

        # Find callers if possible
        callers = []
        if target_symbol != "unknown":
            callers = self._find_callers(repo_root, target_symbol)

        rendered = (
            f"[evidence packet: {target_symbol} @ {target_file}:{target_line}]\n"
            f"slice excerpt:\n{slice_excerpt}\n"
            f"callers: {', '.join(callers)}"
        )
        tokens = estimate_tokens(rendered)

        return EvidencePacket(
            target_file=target_file,
            target_line=target_line,
            target_symbol=target_symbol,
            traceback_summary=traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "",
            slice_excerpt=slice_excerpt,
            callers=callers,
            tokens=tokens,
        )

    def localize_symbol(self, repo_root: Path, symbol_name: str) -> SymbolPacket:
        """Find definition, signature, and callers of a symbol."""
        def_file = "unknown"
        start_line = 0
        end_line = 0
        sig = f"def {symbol_name}(...)"

        for py_file in repo_root.rglob("*.py"):
            if ".git" in py_file.parts or ".venv" in py_file.parts:
                continue
            code = py_file.read_text(encoding="utf-8", errors="ignore")
            if f"def {symbol_name}" in code or f"class {symbol_name}" in code:
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
                            def_file = str(py_file.relative_to(repo_root))
                            start_line = node.lineno
                            end_line = getattr(node, "end_lineno", node.lineno)
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                args = [a.arg for a in node.args.args]
                                sig = f"def {symbol_name}({', '.join(args)})"
                            break
                except Exception:
                    continue
            if def_file != "unknown":
                break

        callers = self._find_callers(repo_root, symbol_name)
        rendered = f"[symbol packet: {symbol_name}]\nlocation: {def_file}:{start_line}-{end_line}\nsignature: {sig}\ncallers: {', '.join(callers)}"
        tokens = estimate_tokens(rendered)

        return SymbolPacket(
            symbol=symbol_name,
            file_path=def_file,
            line_start=start_line,
            line_end=end_line,
            signature=sig,
            callers=callers,
            tokens=tokens,
        )

    def assess_patch(self, old_code: str, new_code: str, symbol_name: str) -> PatchAssessment:
        """Assess whether a patch modifies only function body or changes public interface."""
        try:
            old_tree = ast.parse(old_code)
            new_tree = ast.parse(new_code)
            old_node = next(
                (n for n in ast.walk(old_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol_name),
                None,
            )
            new_node = next(
                (n for n in ast.walk(new_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol_name),
                None,
            )

            if old_node and new_node:
                old_args = [a.arg for a in old_node.args.args]
                new_args = [a.arg for a in new_node.args.args]
                if old_args == new_args:
                    return PatchAssessment(
                        change_type="body_only",
                        symbols_modified=[symbol_name],
                        interface_compatible=True,
                    )
        except Exception:
            pass

        return PatchAssessment(
            change_type="interface_changed",
            symbols_modified=[symbol_name],
            interface_compatible=False,
        )

    def _find_callers(self, repo_root: Path, symbol_name: str) -> list[str]:
        callers = []
        call_pattern = re.compile(rf"\b{re.escape(symbol_name)}\s*\(")
        for py_file in repo_root.rglob("*.py"):
            if ".git" in py_file.parts or ".venv" in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if call_pattern.search(content):
                    rel = str(py_file.relative_to(repo_root))
                    callers.append(rel)
            except Exception:
                continue
        return callers

    def inspect_test_failure(self, repo_root: Path, failure_text: str) -> EvidencePacket:
        """Detailed test failure decomposition, frame parsing, and caller bundling."""
        return self.investigate_failure(repo_root, failure_text)

    def trace_value_origin(self, repo_root: Path, file_path: str, var_name: str) -> dict[str, Any]:
        """Find AST assignments, parameter bindings, or returns defining var_name."""
        target_file = (repo_root / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path)
        if not target_file.exists():
            return {"file": file_path, "origins": [], "error": "file not found"}

        code = target_file.read_text(encoding="utf-8", errors="ignore")
        origins = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if (isinstance(t, ast.Name) and t.id == var_name) or (isinstance(t, ast.Attribute) and t.attr == var_name):
                            origins.append({
                                "line": node.lineno,
                                "type": "assignment",
                                "target": getattr(t, "id", getattr(t, "attr", var_name)),
                            })
                elif isinstance(node, ast.AnnAssign):
                    t = node.target
                    if (isinstance(t, ast.Name) and t.id == var_name) or (isinstance(t, ast.Attribute) and t.attr == var_name):
                        origins.append({
                            "line": node.lineno,
                            "type": "annotated_assignment",
                            "target": getattr(t, "id", getattr(t, "attr", var_name)),
                        })
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args:
                        if arg.arg == var_name:
                            origins.append({
                                "line": node.lineno,
                                "type": "parameter",
                                "function": node.name,
                            })
        except Exception as e:
            return {"file": file_path, "origins": origins, "error": str(e)}

        return {"file": file_path, "variable": var_name, "origins": origins}

    def find_state_writers(self, repo_root: Path, attr_name: str) -> list[dict[str, Any]]:
        """Search repository ASTs for assignments mutating attr_name (e.g. self.x = y)."""
        writers = []
        for py_file in repo_root.rglob("*.py"):
            if ".git" in py_file.parts or ".venv" in py_file.parts:
                continue
            try:
                code = py_file.read_text(encoding="utf-8", errors="ignore")
                if f".{attr_name}" not in code:
                    continue
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if isinstance(t, ast.Attribute) and t.attr == attr_name:
                                rel = str(py_file.relative_to(repo_root))
                                writers.append({
                                    "file": rel,
                                    "line": node.lineno,
                                    "attribute": attr_name,
                                })
            except Exception:
                continue
        return writers

    def find_change_ripple(self, repo_root: Path, symbol_name: str) -> dict[str, Any]:
        """Compute caller graph ripple effect (direct callers + second-order callers)."""
        direct_callers = self._find_callers(repo_root, symbol_name)
        ripple_callers: set[str] = set()

        for c_file in direct_callers:
            c_path = repo_root / c_file
            if c_path.exists():
                try:
                    tree = ast.parse(c_path.read_text(encoding="utf-8", errors="ignore"))
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            # Find callers of this enclosing function
                            second_level = self._find_callers(repo_root, node.name)
                            for s in second_level:
                                if s != c_file:
                                    ripple_callers.add(s)
                except Exception:
                    continue

        return {
            "symbol": symbol_name,
            "direct_callers": direct_callers,
            "ripple_callers": sorted(list(ripple_callers)),
            "impact_count": len(direct_callers) + len(ripple_callers),
        }

    def discover_test_command(self, repo_root: Path, target_path: str) -> str:
        """Find the corresponding test file and command for a source file."""
        stem = Path(target_path).stem
        candidates = [
            f"tests/test_{stem}.py",
            f"test/test_{stem}.py",
            f"tests/{stem}_test.py",
            f"tests/acceptance/test_{stem}.py",
        ]
        for c in candidates:
            if (repo_root / c).exists():
                return f"pytest {c} -q"

        # Search for any test mentioning the stem
        for p in repo_root.rglob(f"*{stem}*.py"):
            if "test" in p.name:
                rel = str(p.relative_to(repo_root))
                return f"pytest {rel} -q"

        return "pytest -q"

    def analyze_import_failure(self, repo_root: Path, error_message: str) -> dict[str, Any]:
        """Diagnose import failure and suggest module resolution."""
        mod_match = re.search(r"No module named '([^']+)'", error_message)
        name_match = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_message)

        missing_module = mod_match.group(1) if mod_match else ""
        missing_symbol = name_match.group(1) if name_match else ""
        from_module = name_match.group(2) if name_match else ""

        suggestions = []
        if missing_module:
            rel_candidate = missing_module.replace(".", "/") + ".py"
            dir_candidate = missing_module.replace(".", "/") + "/__init__.py"
            for py_file in repo_root.rglob("*.py"):
                if py_file.name == Path(rel_candidate).name:
                    suggestions.append(f"found file at {py_file.relative_to(repo_root)}; check PYTHONPATH or package prefix")

        return {
            "missing_module": missing_module,
            "missing_symbol": missing_symbol,
            "from_module": from_module,
            "suggestions": suggestions,
        }

    def compare_failure_delta(self, before_output: str, after_output: str) -> dict[str, Any]:
        """Compare failure signatures between two test runs."""
        def extract_failures(text: str) -> set[str]:
            fails = set()
            for line in text.splitlines():
                if line.strip().startswith("FAILED "):
                    fails.add(line.replace("FAILED ", "").split(" - ")[0].strip())
            return fails

        before_fails = extract_failures(before_output)
        after_fails = extract_failures(after_output)

        resolved = sorted(list(before_fails - after_fails))
        regressions = sorted(list(after_fails - before_fails))
        persistent = sorted(list(before_fails & after_fails))

        return {
            "resolved": resolved,
            "regressions": regressions,
            "persistent": persistent,
        }

    def verify_public_api(self, repo_root: Path, module_path: str) -> dict[str, Any]:
        """Verify that symbols listed in __all__ are defined and exported."""
        target_file = (repo_root / module_path).resolve() if not Path(module_path).is_absolute() else Path(module_path)
        if not target_file.exists():
            return {"module": module_path, "valid": False, "error": "file not found"}

        code = target_file.read_text(encoding="utf-8", errors="ignore")
        defined_symbols = set()
        exported_symbols = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined_symbols.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        exported_symbols.add(elt.value)
        except Exception as e:
            return {"module": module_path, "valid": False, "error": str(e)}

        missing = sorted(list(exported_symbols - defined_symbols))
        return {
            "module": module_path,
            "exported_count": len(exported_symbols),
            "missing_exports": missing,
            "valid": len(missing) == 0,
        }

    def find_similar_fix(self, repo_root: Path, pattern: str) -> list[dict[str, Any]]:
        """Search existing code for similar exception handling or defensive guards."""
        matches = []
        pat = re.compile(rf"\b{re.escape(pattern)}\b")
        for py_file in repo_root.rglob("*.py"):
            if ".git" in py_file.parts or ".venv" in py_file.parts:
                continue
            try:
                lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                for i, line in enumerate(lines, 1):
                    if pat.search(line):
                        rel = str(py_file.relative_to(repo_root))
                        matches.append({
                            "file": rel,
                            "line": i,
                            "content": line.strip(),
                        })
                        if len(matches) >= 5:
                            return matches
            except Exception:
                continue
        return matches

    def proactive_diagnose_failure(self, repo_root: Path, raw_test_output: str) -> str:
        """Automatically synthesize failure investigation without frontier turn cost."""
        lines = [line.strip() for line in raw_test_output.splitlines() if line.strip()]
        failing_tests = [l.replace("FAILED ", "").split(" - ")[0].strip() for l in lines if l.startswith("FAILED ")]
        primary = failing_tests[0] if failing_tests else "unknown"

        ev = self.investigate_failure(repo_root, raw_test_output)
        out_lines = [
            f"[proactive failure diagnosis]",
            f"primary failure: {primary}",
            f"target symbol: {ev.target_symbol} @ {ev.target_file}:{ev.target_line}",
        ]
        if ev.callers:
            out_lines.append(f"callers: {', '.join(ev.callers[:4])}")
        if ev.slice_excerpt:
            first_three = "\n".join(ev.slice_excerpt.splitlines()[:3])
            out_lines.append(f"target slice:\n{first_three}")
        return "\n".join(out_lines)

    def failure_analysis_transaction(
        self,
        repo_root: Path,
        failing_test: str,
        traceback_text: str,
        current_diff: str = "",
    ) -> dict[str, Any]:
        """Compile 6 deterministic static steps into a single zero-turn transaction."""
        ev = self.investigate_failure(repo_root, traceback_text)
        sym_packet = (
            self.localize_symbol(repo_root, ev.target_symbol)
            if ev.target_symbol != "unknown"
            else None
        )
        writes = (
            self.state_writers(repo_root, ev.target_symbol)
            if ev.target_symbol != "unknown"
            else []
        )
        ripple = (
            self.change_ripple(repo_root, current_diff)
            if current_diff
            else []
        )

        ambiguity_lines = [
            f"[failure analysis transaction]",
            f"test: {failing_test}",
            f"failing node: {ev.target_symbol} ({ev.target_file}:{ev.target_line})",
        ]
        if ev.callers:
            ambiguity_lines.append(f"callers ({len(ev.callers)}): {', '.join(ev.callers[:3])}")
        if writes:
            ambiguity_lines.append(f"writers ({len(writes)}): {', '.join(w.writer_function for w in writes[:2])}")
        if ripple:
            ambiguity_lines.append(f"affected dependents: {', '.join(ripple[:3])}")

        return {
            "test": failing_test,
            "target_symbol": ev.target_symbol,
            "target_file": ev.target_file,
            "target_line": ev.target_line,
            "callers": ev.callers,
            "writers_count": len(writes),
            "ripple_count": len(ripple),
            "summary": "\n".join(ambiguity_lines),
            "tokens": estimate_tokens("\n".join(ambiguity_lines)),
        }


def compute_retrieval_consensus(retriever_nominations: list[set[str]]) -> float:
    """Compute consensus C in [0.0, 1.0] across multiple independent retrievers."""
    valid_sets = [s for s in retriever_nominations if s]
    k = len(valid_sets)
    if k <= 1:
        return 1.0 if k == 1 else 0.0

    total_pairs = 0
    sum_jaccard = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            s1 = valid_sets[i]
            s2 = valid_sets[j]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = (intersection / float(union)) if union > 0 else 0.0
            sum_jaccard += jaccard
            total_pairs += 1

    return round(sum_jaccard / float(max(1, total_pairs)), 4)


def recommended_packet_budget(consensus: float) -> tuple[int, str]:
    """Map consensus score to adaptive token budget and retrieval policy."""
    if consensus >= 0.75:
        return 350, "high_consensus_narrow_packet"
    if consensus >= 0.40:
        return 650, "moderate_consensus_extended_packet"
    if consensus >= 0.20:
        return 1200, "low_consensus_broad_source"
    return 0, "disagreement_shell_search_first"


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


class SemanticCoprocessor:
    """Local coprocessor orchestrating deterministic inspections without frontier turns."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root

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

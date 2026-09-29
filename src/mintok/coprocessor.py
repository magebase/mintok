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

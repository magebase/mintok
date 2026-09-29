"""Repository Execution Profiles and complexity priors for MinTok 3.0.

Provides durable, cached repository onboarding profiles capturing package managers,
test commands, monorepo package layout, and complexity priors. Eliminates redundant
frontier exploration across repeated tasks on the same codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mintok.tokens import estimate_tokens


@dataclass
class RepoProfile:
    """Execution profile and topological prior for a codebase."""

    repo_name: str
    language: str = "python"
    package_manager: str = "unknown"
    test_runner: str = "pytest"
    test_command: str = "pytest"
    packages: list[str] = field(default_factory=list)
    key_directories: list[str] = field(default_factory=list)
    complexity_score: float = 0.0
    recommended_strategy: str = "compressed-first"
    build_commands: list[str] = field(default_factory=list)
    workspace_topology: dict[str, list[str]] = field(default_factory=dict)
    generated_directories: list[str] = field(default_factory=list)
    ci_conventions: list[str] = field(default_factory=list)
    test_symbol_map: dict[str, list[str]] = field(default_factory=dict)
    co_change_graph: dict[str, list[str]] = field(default_factory=dict)
    failure_modes: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def render_context(self) -> str:
        """Render a compact ~100-token execution profile for model context."""
        lines = [
            f"[repository profile: {self.repo_name}]",
            f"language: {self.language}",
            f"package_manager: {self.package_manager}",
            f"test_command: {self.test_command}",
            f"packages: {', '.join(self.packages[:8])}",
            f"key_directories: {', '.join(self.key_directories[:6])}",
            f"complexity: {self.complexity_score:.2f} (strategy: {self.recommended_strategy})",
        ]
        if self.build_commands:
            lines.append(f"build_command: {self.build_commands[0]}")
        return "\n".join(lines)

    def record_test_symbol(self, test: str, symbol: str) -> None:
        """Associate a test with a tested symbol."""
        if test not in self.test_symbol_map:
            self.test_symbol_map[test] = []
        if symbol not in self.test_symbol_map[test]:
            self.test_symbol_map[test].append(symbol)

    def record_co_change(self, file_a: str, file_b: str) -> None:
        """Record historical co-change relationship between two files."""
        if file_a not in self.co_change_graph:
            self.co_change_graph[file_a] = []
        if file_b not in self.co_change_graph[file_a]:
            self.co_change_graph[file_a].append(file_b)

        if file_b not in self.co_change_graph:
            self.co_change_graph[file_b] = []
        if file_a not in self.co_change_graph[file_b]:
            self.co_change_graph[file_b].append(file_a)

    def record_failure_mode(self, signature: str, note: str) -> None:
        """Record a recurring failure signature and its resolution note."""
        self.failure_modes.append({
            "signature": signature,
            "note": note,
            "recorded_at": Path(__file__).stat().st_mtime,
        })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoProfile:
        # Handle backward compatibility for any missing fields
        defaults = {
            "build_commands": [],
            "workspace_topology": {},
            "generated_directories": [],
            "ci_conventions": [],
            "test_symbol_map": {},
            "co_change_graph": {},
            "failure_modes": [],
            "metrics": {},
        }
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RepoProfile:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def get_or_create_repo_profile(root: Path, cache_dir: Path | None = None) -> RepoProfile:
    """Retrieve durable cached profile for a repository or scan and cache it."""
    cache_base = cache_dir or (root / ".mintok")
    profile_file = cache_base / "profile.json"
    if profile_file.exists():
        try:
            return RepoProfile.load(profile_file)
        except Exception:
            pass

    profile = scan_repo_profile(root)
    try:
        profile.save(profile_file)
    except Exception:
        pass
    return profile


def scan_repo_profile(root: Path, repo_name: str | None = None) -> RepoProfile:
    """Scan a repository directory and construct its RepoProfile."""
    name = repo_name or root.name
    files: list[Path] = []
    dirs: set[str] = set()
    packages: list[str] = []
    max_depth = 0

    for p in root.rglob("*"):
        if ".git" in p.parts or "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            max_depth = depth
        if p.is_dir():
            dirs.add(str(rel))
            if (p / "__init__.py").exists():
                packages.append(str(rel).replace("/", "."))
        elif p.is_file():
            files.append(rel)

    # Detect package manager
    pkg_mgr = "unknown"
    rel_files_str = {str(f) for f in files}
    if "poetry.lock" in rel_files_str or any("tool.poetry" in p.read_text(errors="ignore") for p in [root / "pyproject.toml"] if p.exists()):
        pkg_mgr = "poetry"
    elif "pyproject.toml" in rel_files_str:
        pkg_mgr = "pyproject"
    elif "setup.py" in rel_files_str or "setup.cfg" in rel_files_str:
        pkg_mgr = "setuptools"
    elif "Pipfile" in rel_files_str:
        pkg_mgr = "pipenv"
    elif "requirements.txt" in rel_files_str:
        pkg_mgr = "pip"

    # Detect test runner & command
    test_runner = "pytest"
    test_cmd = "pytest -q"
    if any("unittest" in str(f) for f in files) and not any("pytest" in str(f) for f in files):
        test_runner = "unittest"
        test_cmd = "python -m unittest discover"

    # Identify key directories
    key_dirs = sorted(
        [d for d in dirs if d in ("src", "tests", "test", "pkg") or d.startswith("src/") or ("tests" in d and len(d.split("/")) <= 2)],
        key=lambda x: (len(x.split("/")), x),
    )
    if not key_dirs:
        key_dirs = sorted(list(dirs)[:4])

    file_count = len(files)
    package_count = len(packages)

    # Complexity score: [0.0, 1.0]
    depth_factor = min(1.0, max_depth / 6.0)
    pkg_factor = min(1.0, package_count / 5.0)
    file_factor = min(1.0, file_count / 150.0)
    complexity = 0.35 * depth_factor + 0.45 * pkg_factor + 0.20 * file_factor

    strategy = "virtualized-shell" if (complexity >= 0.50 or package_count >= 4 or max_depth >= 5) else "compressed-first"

    build_cmds = []
    if "pyproject.toml" in rel_files_str:
        build_cmds.append("pip install -e .")
    elif "setup.py" in rel_files_str:
        build_cmds.append("python setup.py build")
    elif "Makefile" in rel_files_str:
        build_cmds.append("make")

    generated_dirs = [d for d in dirs if any(d == g or d.endswith(f"/{g}") for g in ("build", "dist", ".tox", "target"))]

    ci = []
    if any(".github" in d for d in dirs):
        ci.append("github-actions")
    if any(".circleci" in d for d in dirs):
        ci.append("circleci")
    if "tox.ini" in rel_files_str:
        ci.append("tox")

    topology: dict[str, list[str]] = {}
    for pkg in packages:
        top = pkg.split(".")[0]
        if top not in topology:
            topology[top] = []
        topology[top].append(pkg)

    metrics = {
        "file_count": file_count,
        "package_count": package_count,
        "max_depth": max_depth,
    }

    return RepoProfile(
        repo_name=name,
        language="python",
        package_manager=pkg_mgr,
        test_runner=test_runner,
        test_command=test_cmd,
        packages=packages,
        key_directories=key_dirs,
        complexity_score=complexity,
        recommended_strategy=strategy,
        build_commands=build_cmds,
        workspace_topology=topology,
        generated_directories=generated_dirs,
        ci_conventions=ci,
        metrics=metrics,
    )

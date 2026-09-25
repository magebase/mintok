"""End-to-end benchmark task set: real repository tasks with automated checkers.

Every task runs on a fresh copy of this repository at a fixed commit and is
scored by (a) the repository's own test suite passing and (b) a task-specific
behavioral checker. Checkers are plain Python executed with ``root`` bound to
the task copy and ``sys.path`` pointing at the copy's ``src``; each must set
``ok`` (bool) and optionally ``detail`` (str).

Task classes span parameter propagation, renames, new symbols, CLI work,
behavior changes with suite updates, interface-preserving refactors,
documentation-only edits, error handling, multi-file moves, class-level and
decorator changes, and cleanup - deliberately not shaped around the ABI's
strengths.
"""

from __future__ import annotations

TASKS: list[dict] = [
    {
        "id": "add-param-renderfact",
        "klass": "add_parameter",
        "instruction": (
            "Add a keyword parameter `include_provenance: bool = False` to "
            "`render_fact` in src/mintok/ir.py. When True, the rendered line must "
            "end with ` @{provenance}` (e.g. 'calls x 1.00 @python-ast'). Default "
            "rendering must stay byte-identical. Update all callers in src/ so the "
            "full test suite passes."
        ),
        "check": """
from mintok.ir import Fact, render_fact
f = Fact("a:b", "calls", "c:d", 0.5)
assert render_fact(f) == "calls c:d 0.50"
assert render_fact(f, include_provenance=True) == "calls c:d 0.50 @python-ast"
ok = True
""",
    },
    {
        "id": "add-param-factsfor",
        "klass": "add_parameter",
        "instruction": (
            "Add a `min_confidence: float = 0.0` parameter to `ProgramIR.facts_for` "
            "in src/mintok/ir.py; it must only return facts whose confidence is "
            ">= min_confidence. Existing call sites keep working unchanged."
        ),
        "check": """
from mintok.ir import Fact, ProgramIR
ir = ProgramIR()
ir.facts.append(Fact("s", "calls", "t", 0.25))
assert len(ir.facts_for("s")) == 1
assert ir.facts_for("s", min_confidence=0.5) == []
assert [f.object for f in ir.facts_for("s", min_confidence=0.25)] == ["t"]
ok = True
""",
    },
    {
        "id": "add-param-estimate",
        "klass": "add_parameter",
        "instruction": (
            "Make `estimate_tokens` in src/mintok/tokens.py accept an optional "
            "`minimum: int = 0` parameter that floors the estimate (after "
            "rounding, before returning; 0 text still estimates 0)."
        ),
        "check": """
from mintok.tokens import estimate_tokens
assert estimate_tokens("") == 0
assert estimate_tokens("x" * 8, minimum=10) == 10
assert estimate_tokens("x" * 8) == 2
ok = True
""",
    },
    {
        "id": "add-cli-verify",
        "klass": "add_parameter",
        "instruction": (
            "Add a `mintok verify ROOT -- ARGV...` CLI command in src/mintok/cli.py "
            "that runs the mintok ABI verify op: print PASS or FAIL plus the output "
            "tail, and exit with the command's exit code."
        ),
        "check": """
import contextlib, io
from mintok.cli import main
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = main(["verify", root, "--", "python", "-c", "print('hello lines'); raise SystemExit(0)"])
out = buf.getvalue()
assert out.startswith("PASS"), out
assert code == 0
ok = True
""",
    },
    {
        "id": "rename-stable-hash",
        "klass": "rename",
        "instruction": (
            "Rename the function `stable_hash` in src/mintok/ir.py to "
            "`semantic_hash` and update every import and usage under src/ so the "
            "full test suite passes."
        ),
        "check": """
import mintok.ir as irmod
assert not hasattr(irmod, "stable_hash")
from mintok.ir import semantic_hash
assert semantic_hash("a", "b") == semantic_hash("a", "b") and len(semantic_hash("a")) == 16
ok = True
""",
    },
    {
        "id": "rename-cached-facts",
        "klass": "rename",
        "instruction": (
            "Rename the `facts` method of `FactCache` in src/mintok/cache.py to "
            "`cached_facts` and update every caller under src/ and tests/ so the "
            "full test suite passes."
        ),
        "check": """
from mintok.cache import FactCache
assert not hasattr(FactCache, "facts") or "facts" not in FactCache.__dict__
assert "cached_facts" in FactCache.__dict__
ok = True
""",
    },
    {
        "id": "rename-savings-pct",
        "klass": "rename",
        "instruction": (
            "Rename the `saved_pct` property of `BenchReport` in src/mintok/bench.py "
            "to `savings_pct` and update every usage under src/ so the full test "
            "suite passes."
        ),
        "check": """
from mintok.bench import BenchReport
assert hasattr(BenchReport, "savings_pct")
assert not hasattr(BenchReport, "saved_pct")
ok = True
""",
    },
    {
        "id": "rename-dotted-module",
        "klass": "rename",
        "instruction": (
            "Rename the function `module_name` in src/mintok/compiler/python.py to "
            "`dotted_module` and update every usage under src/ and tests/ so the "
            "full test suite passes."
        ),
        "check": """
import mintok.compiler.python as cpm
assert not hasattr(cpm, "module_name")
from mintok.compiler.python import dotted_module
from pathlib import Path
assert dotted_module(Path("src/mintok/__init__.py")) == "mintok"
ok = True
""",
    },
    {
        "id": "new-cli-symbols",
        "klass": "new_symbol",
        "instruction": (
            "Add a `mintok symbols ROOT` CLI command that prints one line per "
            "symbol, sorted by id, in the format `<id> <kind>`, and exits 0. "
            "Diagnostics go to stderr as usual."
        ),
        "check": """
import contextlib, io
from mintok.cli import main
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = main(["symbols", root])
lines = buf.getvalue().splitlines()
assert code == 0 and lines and all(" " in ln for ln in lines)
assert any(ln.split(" ")[0].endswith("abi:AgentABI") for ln in lines)
ok = True
""",
    },
    {
        "id": "new-ir-symbols-of-kind",
        "klass": "new_symbol",
        "instruction": (
            "Add a method `symbols_of_kind(self, kind: str)` to `ProgramIR` in "
            "src/mintok/ir.py returning the symbols of that kind sorted by id."
        ),
        "check": """
from mintok.ir import ProgramIR
ir = ProgramIR()
assert ir.symbols_of_kind("function") == []
ids = [s.id for s in ir.symbols_of_kind("class")]
assert ids == sorted(ids)
ok = True
""",
    },
    {
        "id": "new-pack-is-empty",
        "klass": "new_symbol",
        "instruction": (
            "Add an `is_empty` property to `RelearnPack` in src/mintok/pack.py: "
            "True exactly when the pack reports no changes to relearn."
        ),
        "check": """
from mintok.pack import RelearnPack
assert RelearnPack(changes=()).is_empty is True
assert RelearnPack(changes=(), renderings={}, omitted_body_only=1).is_empty is False
ok = True
""",
    },
    {
        "id": "new-cli-ops-flag",
        "klass": "new_symbol",
        "instruction": (
            "Add a `--ops` flag to the root `mintok` command that prints the valid "
            "query ops (one per line, from OPEN_QUERY_OPS) and exits 0."
        ),
        "check": """
import contextlib, io
from mintok.cli import build_parser, main
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = main(["--ops"])
out = buf.getvalue().splitlines()
assert code == 0 and "find" in out and "symbol" in out and "slice" not in out
ok = True
""",
    },
    {
        "id": "behavior-tail-10",
        "klass": "behavior_change",
        "instruction": (
            "Change `AgentABI.verify` so the default tail is 10 lines instead of 20, "
            "and update the test suite (including any Gherkin scenario pinned to 20) "
            "so the full suite passes with the new default."
        ),
        "check": """
import inspect
from mintok.abi import AgentABI
sig = inspect.signature(AgentABI.verify)
assert sig.parameters["tail_lines"].default == 10
ok = True
""",
    },
    {
        "id": "behavior-estimate-chars3",
        "klass": "behavior_change",
        "instruction": (
            "Change the token estimator in src/mintok/tokens.py from chars/4 to "
            "chars/3 (ceil; empty text still 0). The full test suite must pass "
            "afterwards; if a scenario pins an exact budget, adjust the budget, "
            "never the assertion style."
        ),
        "check": """
from mintok.tokens import estimate_tokens
assert estimate_tokens("x" * 12) == 4
assert estimate_tokens("x" * 10) == 4
assert estimate_tokens("") == 0
ok = True
""",
    },
    {
        "id": "behavior-find-casesensitive",
        "klass": "behavior_change",
        "instruction": (
            "Make `AgentABI.find_symbols` match case-sensitively instead of "
            "case-insensitively, document that in the tool surface description, and "
            "make sure the full test suite passes."
        ),
        "check": """
import textwrap, tempfile
from pathlib import Path
from mintok.abi import AgentABI
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "m.py").write_text("def refund():\\n    return True\\n")
    abi = AgentABI(td)
    assert [s.id for s in abi.find_symbols("refund")] == ["m:refund"]
    assert abi.find_symbols("REFUND") == []
ok = True
""",
    },
    {
        "id": "refactor-resolve-call",
        "klass": "interface_preserving_refactor",
        "instruction": (
            "Refactor `_resolve_call` in src/mintok/compiler/python.py into smaller "
            "helper functions without changing any behavior. The full test suite "
            "must pass and the module must still produce identical facts for the "
            "same inputs."
        ),
        "check": """
import textwrap, tempfile
from pathlib import Path
from mintok.compiler import compile_repository
src = (
    "class G:\\n"
    "    def ping(self):\\n"
    "        return True\\n"
    "    def pong(self):\\n"
    "        return self.ping()\\n"
)
with tempfile.TemporaryDirectory() as td:
    (Path(td) / "m.py").write_text(src)
    ir = compile_repository(td)
    calls = {f.object for f in ir.facts_for("m:G.pong", "calls")}
    assert calls == {"m:G.ping"}, calls
ok = True
""",
    },
    {
        "id": "refactor-indent-helper",
        "klass": "interface_preserving_refactor",
        "instruction": (
            "Extract the indentation-detection logic inside `AgentABI.change` in "
            "src/mintok/abi.py into a module-level helper `_indent_of(line: str) -> str` "
            "and use it from `change`. Behavior must be identical; suite passes."
        ),
        "check": """
from mintok.abi import _indent_of
assert _indent_of("    x = 1") == "    "
assert _indent_of("no indent") == ""
assert _indent_of("\\ttabbed") == "\\t"
ok = True
""",
    },
    {
        "id": "docstring-tokens-module",
        "klass": "docstring_only",
        "instruction": (
            "Write a comprehensive module docstring for src/mintok/tokens.py "
            "explaining the estimator's role in cost accounting and why it is "
            "pluggable (swap per model backend when measured)."
        ),
        "check": """
import ast
from pathlib import Path
tree = ast.parse((Path(root) / "src/mintok/tokens.py").read_text())
doc = ast.get_docstring(tree) or ""
assert len(doc) > 100 and "token" in doc.lower() and "pluggable" in doc.lower()
ok = True
""",
    },
    {
        "id": "docstring-factcache",
        "klass": "docstring_only",
        "instruction": (
            "Add a docstring to every public method of `FactCache` in "
            "src/mintok/cache.py that lacks one (from_ir, is_valid, facts, refresh). "
            "Behavior unchanged; suite passes."
        ),
        "check": """
import inspect
from mintok.cache import FactCache
for name in ("from_ir", "is_valid", "facts", "refresh"):
    method = getattr(FactCache, name)
    assert method.__doc__, name
ok = True
""",
    },
    {
        "id": "comment-skipdirs",
        "klass": "comment_only",
        "instruction": (
            "Add an inline comment next to SKIP_DIRS in src/mintok/compiler/python.py "
            "explaining what the list is for and why each entry is skipped during "
            "indexing."
        ),
        "check": """
from pathlib import Path
text = (Path(root) / "src/mintok/compiler/python.py").read_text()
lines = [ln for ln in text.splitlines() if ln.strip().startswith("#") and "SKIP_DIRS" not in ln]
assert any("skip" in ln.lower() or "index" in ln.lower() for ln in lines), lines[:5]
ok = True
""",
    },
    {
        "id": "errors-compile-missing-root",
        "klass": "error_handling",
        "instruction": (
            "Make `mintok compile` fail cleanly when the root path does not exist: "
            "print a one-line error to stderr and exit with code 2, with no "
            "traceback. All existing behavior for valid roots must be unchanged."
        ),
        "check": """
import contextlib, io
from mintok.cli import main
err = io.StringIO()
out = io.StringIO()
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
    code = main(["compile", "/nonexistent/path/for/benchmark"])
assert code == 2, code
assert "traceback" not in err.getvalue().lower()
assert err.getvalue().strip() != ""
ok = True
""",
    },
    {
        "id": "errors-query-unknown-op",
        "klass": "error_handling",
        "instruction": (
            "Make `AgentABI.query` raise a KeyError whose message lists the valid "
            "ops (include the exact text 'valid ops:') when the op is unknown."
        ),
        "check": """
import textwrap, tempfile
from pathlib import Path
from mintok.abi import AgentABI
with tempfile.TemporaryDirectory() as td:
    abi = AgentABI(td)
    try:
        abi.query("explode", "x")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "valid ops:" in str(exc), exc
ok = True
""",
    },
    {
        "id": "multi-move-stable-hash",
        "klass": "multi_file",
        "instruction": (
            "Move the `stable_hash` function from src/mintok/ir.py into a new module "
            "src/mintok/hashing.py. `from mintok.ir import stable_hash` must keep "
            "working (re-export), all internal uses under src/ should import from "
            "the new module, and the full test suite must pass. You may create new files."
        ),
        "check": """
import subprocess, sys
code = "from mintok.ir import stable_hash as a; from mintok.hashing import stable_hash as b; assert a is b"
res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
assert res.returncode == 0, res.stderr
ok = True
""",
    },
    {
        "id": "multi-errors-module",
        "klass": "multi_file",
        "instruction": (
            "Create src/mintok/errors.py defining `class MintokError(Exception)`, "
            "make `ChangeRejected` in src/mintok/abi.py inherit from it (class-level "
            "edit), and keep `from mintok.abi import ChangeRejected` working. Suite "
            "passes. You may create new files."
        ),
        "check": """
from mintok.errors import MintokError
from mintok.abi import ChangeRejected
assert issubclass(ChangeRejected, MintokError)
ok = True
""",
    },
    {
        "id": "class-change-factcache",
        "klass": "class_level_change",
        "instruction": (
            "Replace the entire `FactCache` class in src/mintok/cache.py with a "
            "version that keeps all current behavior and additionally supports "
            "`len(cache)` (returning the number of cached symbols). Suite passes."
        ),
        "check": """
from mintok.cache import FactCache
cache = FactCache()
assert len(cache) == 0
from mintok.compiler import compile_repository
ir = compile_repository(root)
cache = FactCache.from_ir(ir)
assert len(cache) == len(ir.symbols) > 0
ok = True
""",
    },
    {
        "id": "class-change-programir",
        "klass": "class_level_change",
        "instruction": (
            "Replace the entire `ProgramIR` class in src/mintok/ir.py with a version "
            "that keeps all current behavior and additionally exposes a "
            "`fact_count` property (number of facts). Suite passes."
        ),
        "check": """
from mintok.ir import ProgramIR
ir = ProgramIR()
assert ir.fact_count == 0
ir.facts.append(type(ir.facts[0])("s", "p", "o"))
assert ir.fact_count == 1
ok = True
""",
    },
    {
        "id": "decorator-lrucache",
        "klass": "class_level_change",
        "instruction": (
            "Add `import functools` and decorate `tool_surface_json` in "
            "src/mintok/abi.py with `@functools.lru_cache(maxsize=1)` (its result is "
            "constant). Suite passes and repeated calls return the same object."
        ),
        "check": """
from mintok.abi import tool_surface_json
assert tool_surface_json() is tool_surface_json()
assert '"name":"query"' in tool_surface_json()
ok = True
""",
    },
    {
        "id": "cleanup-kinds",
        "klass": "cleanup",
        "instruction": (
            "Delete the unused `KINDS` constant from src/mintok/diff.py (verify it "
            "is truly unused under src/ and tests/ first) so the full test suite "
            "passes."
        ),
        "check": """
import mintok.diff as d
assert not hasattr(d, "KINDS")
from mintok.diff import diff_ir, render_changes, SemanticChange
ok = True
""",
    },
    {
        "id": "cli-diff-quiet",
        "klass": "cli",
        "instruction": (
            "Add a `--quiet` flag to `mintok diff`: when set, print nothing and exit "
            "1 if any semantic changes exist, else exit 0. Without --quiet, behavior "
            "is unchanged. Suite passes."
        ),
        "check": """
import contextlib, io, textwrap, tempfile
from pathlib import Path
from mintok.cli import main
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "same").mkdir(); (td / "other").mkdir()
    (td / "same" / "m.py").write_text("def f():\\n    return True\\n")
    (td / "other" / "m.py").write_text("def f():\\n    return True\\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["diff", str(td / "same"), str(td / "other"), "--quiet"]) == 0
    (td / "other" / "m.py").write_text("def f():\\n    return None\\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["diff", str(td / "same"), str(td / "other"), "--quiet"]) == 1
ok = True
""",
    },
    {
        "id": "cli-callers-command",
        "klass": "cli",
        "instruction": (
            "Add a `mintok callers ROOT SYMBOL` CLI command that prints one line per "
            "caller of the symbol (`<caller id> <confidence>`), sorted, exiting 0 "
            "even when there are none."
        ),
        "check": """
import contextlib, io, textwrap, tempfile
from pathlib import Path
from mintok.cli import main
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "m.py").write_text("def ping():\\n    return True\\n\\n\\ndef pong():\\n    return ping()\\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["callers", str(td), "m:ping"])
    assert code == 0
    assert buf.getvalue().splitlines() == ["m:pong 1.00"], buf.getvalue()
ok = True
""",
    },
]

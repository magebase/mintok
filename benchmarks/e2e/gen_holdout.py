"""Fresh large-module holdout generator (30-50 tasks across 4 new repos).

Emitted ONCE, then frozen: repos land in fixtures/<repo>, tasks in
tasks_holdout.json, and fingerprints commit before any outcome is
inspected. Task wording is fresh (never used in slicer/router tuning);
the bug shapes come from ten generic templates applied to seeded names.

Subcommands:
  --emit      write fixtures + tasks_holdout.json
  --validate  every check FAILS on pristine copies, the fixture suite
              PASSES pristine, and the reference fix makes each check
              PASS without breaking the suite
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
EMIT_PATH = HERE / "tasks_holdout.json"

PREAMBLE = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.join(root, 'src'))\n"
)


def _c(body: str) -> str:
    return PREAMBLE + textwrap.dedent(body).strip("\n") + "\nok = True\n"


# ---------------------------------------------------------------------------
# Bug templates: (template_id, doc)
# Each instance: pristine body has a seeded gap; REF FIX restores canonical
# behavior. Templates produce (pristine_line, fixed_line) pairs keyed by a
# sentinel in the generated source.
# ---------------------------------------------------------------------------

# strip-family: pristine strips a too-small charset; canonical strips the
# full set. Fix: widen the charset argument.
STRIP_SMALL = "' '"
STRIP_FULL = "' \\t\\n-_.'"

# threshold-family: pristine uses strict >; canonical uses >=.
GT_STRICT = "amount > threshold"
GT_CANON = "amount >= threshold"

# bound-family: pristine clamps to the wrong upper constant.
BOUND_WRONG = "min(value, upper_wrong)"
BOUND_RIGHT = "min(value, upper_right)"

# round-family: pristine truncates; canonical rounds half up.
ROUND_WRONG = "total // count"
ROUND_RIGHT = "(total + count - 1) // count"

# case-family: pristine compares case-sensitively; canonical folds case.
CASE_WRONG = "if needle in text:"
CASE_RIGHT = "if needle.lower() in text.lower():"

# default-family: pristine default is wrong; canonical uses the theme zero.
DEFAULT_WRONG = "start: int = 1"
DEFAULT_RIGHT = "start: int = 0"

# slice-family: pristine drops the last element (off-by-one slice).
SLICE_WRONG = "values[:-1]"
SLICE_RIGHT = "values"

# join-family: pristine joins with a doubled separator.
JOIN_WRONG = 'separator.join(parts) + separator'
JOIN_RIGHT = "separator.join(parts)"

# count-family: pristine counts only exact case; canonical counts all case.
COUNT_WRONG = "text.count(ch)"
COUNT_RIGHT = "text.lower().count(ch.lower())"

# normalize-family: pristine preserves inner whitespace; canonical collapses it.
NORM_WRONG = 'text.strip()'
NORM_RIGHT = '" ".join(text.split())'


TEMPLATES = [
    ("strip", STRIP_SMALL, STRIP_FULL),
    ("threshold", GT_STRICT, GT_CANON),
    ("bound", BOUND_WRONG, BOUND_RIGHT),
    ("round", ROUND_WRONG, ROUND_RIGHT),
    ("case", CASE_WRONG, CASE_RIGHT),
    ("default", DEFAULT_WRONG, DEFAULT_RIGHT),
    ("slice", SLICE_WRONG, SLICE_RIGHT),
    ("join", JOIN_WRONG, JOIN_RIGHT),
    ("count", COUNT_WRONG, COUNT_RIGHT),
    ("normalize", NORM_WRONG, NORM_RIGHT),
]

REPOS = {
    "ledger": {
        "pkg": "ledger",
        "title": "journal and posting utilities",
        "words": ["entry", "posting", "amount", "balance", "memo", "voucher",
                  "ledger", "debit", "credit", "reconcile", "batch", "journal"],
    },
    "inventory": {
        "pkg": "inventory",
        "title": "stock and SKU utilities",
        "words": ["stock", "sku", "crate", "shelf", "pallet", "parcel",
                  "warehouse", "count", "bin", "batch", "label", "reorder"],
    },
    "dispatch": {
        "pkg": "dispatch",
        "title": "courier route and schedule utilities",
        "words": ["route", "courier", "manifest", "stop", "leg", "window",
                  "dispatch", "cargo", "pickup", "depot", "lane", "shift"],
    },
    "metrics": {
        "pkg": "metrics",
        "title": "telemetry formatting and aggregation",
        "words": ["gauge", "sample", "bucket", "series", "interval", "datum",
                  "metric", "window", "quantile", "burst", "span", "tick"],
    },
}


def _family_fn(word: str, template_id: str, idx: int) -> tuple[str, str, str]:
    """One function instance: (source, ref_find, ref_replace).

    The pristine body embeds the template's sentinel bug; the reference fix
    swaps the sentinel line for the canonical line.
    """
    body_by_tpl = {
        "strip": (
            f'def clip_{word}(text: str) -> str:\n'
            f'    """Trim surrounding blankery from {word} text."""\n'
            f'    return text.strip({STRIP_SMALL})\n'
        ),
        "threshold": (
            f'def flags_{word}(amount: int, threshold: int) -> bool:\n'
            f'    """True when the {word} reaches the threshold exactly."""\n'
            f'    return {GT_STRICT}\n'
        ),
        "bound": (
            f'def cap_{word}(value: int, upper_wrong: int, upper_right: int) -> int:\n'
            f'    """Clamp the {word} to its ceiling."""\n'
            f'    return {BOUND_WRONG}\n'
        ),
        "round": (
            f'def share_{word}(total: int, count: int) -> int:\n'
            f'    """Per-unit {word} share, rounding half up."""\n'
            f'    return {ROUND_WRONG}\n'
        ),
        "case": (
            f'def finds_{word}(text: str, needle: str) -> bool:\n'
            f'    """Case-insensitive {word} search."""\n'
            f'    {CASE_WRONG}\n'
            f'        return True\n'
            f'    return False\n'
        ),
        "default": (
            f'def reindex_{word}(items: list, start: int = 1) -> dict:\n'
            f'    """Index {word} items from zero."""\n'
            f'    return {{start + i: item for i, item in enumerate(items)}}\n'
        ),
        "slice": (
            f'def keep_{word}(values: list) -> list:\n'
            f'    """Return the {word} values unchanged."""\n'
            f'    return {SLICE_WRONG}\n'
        ),
        "join": (
            f'def link_{word}(parts: list, separator: str = "-") -> str:\n'
            f'    """Join {word} parts with a single separator."""\n'
            f'    return {JOIN_WRONG}\n'
        ),
        "count": (
            f'def tally_{word}(text: str, ch: str) -> int:\n'
            f'    """Count {word} marker letters in any case."""\n'
            f'    return {COUNT_WRONG}\n'
        ),
        "normalize": (
            f'def tidy_{word}(text: str) -> str:\n'
            f'    """Collapse inner whitespace in {word} text."""\n'
            f'    return {NORM_WRONG}\n'
        ),
    }
    ref_fixes = {
        "strip": (f"return text.strip({STRIP_SMALL})", f"return text.strip({STRIP_FULL})"),
        "threshold": (f"return {GT_STRICT}", f"return {GT_CANON}"),
        "bound": (f"return {BOUND_WRONG}", f"return {BOUND_RIGHT}"),
        "round": (f"return {ROUND_WRONG}", f"return {ROUND_RIGHT}"),
        "case": (CASE_WRONG, CASE_RIGHT),
        "default": (
            f"def reindex_{word}(items: list, start: int = 1) -> dict:",
            f"def reindex_{word}(items: list, start: int = 0) -> dict:",
        ),
        "slice": (f"return {SLICE_WRONG}", f"return {SLICE_RIGHT}"),
        "join": (f"return {JOIN_WRONG}", f"return {JOIN_RIGHT}"),
        "count": (f"return {COUNT_WRONG}", f"return {COUNT_RIGHT}"),
        "normalize": (f"return {NORM_WRONG}", f"return {NORM_RIGHT}"),
    }
    src = body_by_tpl[template_id]
    find, replace = ref_fixes[template_id]
    return src, find, replace


def build_repo(repo: str, spec: dict) -> tuple[str, list[dict]]:
    """Return (large.py source, task dicts) for one fixture repo."""
    pkg = spec["pkg"]
    words = spec["words"]
    sections = []
    tasks: list[dict] = []
    # 10 task families at deterministic offsets, plus filler families.
    for fam, (template_id, _, _) in enumerate(TEMPLATES):
        word = words[fam % len(words)]
        filler_words = [w for w in words if w != word]
        chunk = [f"# -- {word} {template_id} helpers " + "-" * 40]
        ref_pairs = []
        for offset in range(len(words)):
            w = word if offset == 0 else filler_words[(offset - 1) % len(filler_words)]
            fn_src, find, replace = _family_fn(w, template_id, offset)
            chunk.append(fn_src)
            if offset == 0:
                ref_pairs.append((find, replace))
        # a class wrapping the family with helpers
        cls_word = word.capitalize()
        chunk.append(
            f"class {cls_word}Kit:\n"
            f'    """{cls_word} helpers, namespaced for callers."""\n\n'
            f"    def __init__(self, tag: str = '{word}') -> None:\n"
            f"        self.tag = tag\n"
            f"        self.ops: list[str] = []\n\n"
            f"    def apply(self, text: str) -> str:\n"
            f"        self.ops.append('apply')\n"
            f"        return text\n\n"
            f"    def probe(self, text: str) -> bool:\n"
            f"        self.ops.append('probe')\n"
            f"        return bool(text)\n\n"
            f"    def summary(self, items: list[str]) -> str:\n"
            f"        self.ops.append('summary')\n"
            f"        return ', '.join(items)\n\n"
            f"    def validate(self, payload: dict) -> bool:\n"
            f"        self.ops.append('validate')\n"
            f"        return bool(payload)\n\n"
            f"    def format_entry(self, key: str, val: int) -> str:\n"
            f"        self.ops.append('format')\n"
            f'        return f"{{key}}:{{val}}"\n\n'
            f"    def parse_entry(self, raw: str) -> tuple[str, int]:\n"
            f"        self.ops.append('parse')\n"
            f"        if ':' not in raw:\n"
            f"            return raw, 0\n"
            f"        k, v = raw.split(':', 1)\n"
            f"        return k, int(v) if v.isdigit() else 0\n\n"
            f"    def transform(self, values: list[str]) -> list[str]:\n"
            f"        self.ops.append('transform')\n"
            f"        return [v.strip() for v in values if v]\n\n"
            f"    def count_ops(self) -> int:\n"
            f"        return len(self.ops)\n\n"
            f"    def reset(self) -> None:\n"
            f"        self.ops.clear()\n"
        )
        sections.append("\n".join(chunk))

        # Task: target the family's primary function with fresh wording.
        task_spec = TASK_SPECS[template_id].format(pkg=pkg, word=word)
        find, replace = ref_pairs[0]
        check = CHECKS[template_id].replace("{word}", word)
        tasks.append(
            {
                "id": f"{repo}-holdout-{fam + 1:02d}",
                "klass": "large_file_navigation",
                "repo": repo,
                "instruction": task_spec,
                "check": _c(check),
                "ref_find": find,
                "ref_replace": replace,
                "target": f"{word}",
                "template": template_id,
            }
        )
    header = (
        f'"""{spec["pkg"]}.{spec["pkg"]} -- {spec["title"]} mega module.\n\n'
        "Everything in this area accumulated here over the years. All pure,\n"
        "stdlib-only. Some helpers duplicate each other; the duplicates are\n"
        "intentional in the fixture.\n"
        '"""\n\nfrom __future__ import annotations\n'
    )
    preamble_blocks = [
        "class OperationError(Exception):\n    \"\"\"Base exception for domain operations.\"\"\"\n    pass\n",
        "class ValidationError(OperationError):\n    \"\"\"Raised when input validation fails.\"\"\"\n    pass\n",
        "class NotFoundError(OperationError):\n    \"\"\"Raised when a requested entity is missing.\"\"\"\n    pass\n",
        "class ConcurrencyError(OperationError):\n    \"\"\"Raised on version conflict.\"\"\"\n    pass\n",
        f"MODULE_TAG = '{pkg}'\nDEFAULT_PAGE_SIZE = 50\nMAX_CACHE_ENTRIES = 1000\nRETRY_ATTEMPTS = 3\nBACKOFF_FACTOR = 1.5\n",
        f"def clamp_value(val: int, low: int, high: int) -> int:\n"
        f'    """Utility clamp within bounds."""\n'
        f"    return max(low, min(high, val))\n\n"
        f"def split_chunks(items: list, size: int) -> list[list]:\n"
        f'    """Partition items into fixed-size chunks."""\n'
        f"    if size <= 0:\n"
        f"        return [items]\n"
        f"    return [items[i:i + size] for i in range(0, len(items), size)]\n\n"
        f"def merge_mappings(primary: dict, fallback: dict) -> dict:\n"
        f'    """Merge fallback entries into primary without clobbering."""\n'
        f"    merged = dict(fallback)\n"
        f"    merged.update(primary)\n"
        f"    return merged\n",
        f"class {pkg.capitalize()}Registry:\n"
        f'    """Central registry and lookup dispatch for {pkg} components."""\n\n'
        f"    def __init__(self) -> None:\n"
        f"        self._handlers: dict[str, object] = {{}}\n"
        f"        self._metadata: dict[str, str] = {{}}\n\n"
        f"    def register(self, name: str, handler: object) -> None:\n"
        f"        self._handlers[name] = handler\n\n"
        f"    def lookup(self, name: str) -> object:\n"
        f"        if name not in self._handlers:\n"
        f"            raise NotFoundError(f'Component not found: {{name}}')\n"
        f"        return self._handlers[name]\n\n"
        f"    def has_handler(self, name: str) -> bool:\n"
        f"        return name in self._handlers\n\n"
        f"    def clear(self) -> None:\n"
        f"        self._handlers.clear()\n"
        f"        self._metadata.clear()\n",
    ]
    preamble = "\n".join(preamble_blocks)
    return header + "\n" + preamble + "\n\n" + "\n\n".join(sections) + "\n", tasks


TASK_SPECS = {
    "strip": "In src/{pkg}/large.py, clip_{word} leaves stray "
             "tabs, newlines, dashes, underscores or dots on {word} text. Make it treat "
             "text built only from whitespace, dashes, underscores and dots as fully trimmed.",
    "threshold": "In src/{pkg}/large.py, flags_{word} misses {word} items that hit the threshold exactly. "
                 "Reaching the threshold must count as meeting it.",
    "bound": "In src/{pkg}/large.py, cap_{word} clamps to the wrong ceiling argument. Clamp to upper_right, "
             "not upper_wrong, and leave the rest unchanged.",
    "round": "In src/{pkg}/large.py, share_{word} truncates instead of rounding half up. A remainder of one "
             "or more must round the share up.",
    "case": "In src/{pkg}/large.py, finds_{word} misses matches that differ in letter case. Search must be "
            "case-insensitive for {word} text.",
    "default": "In src/{pkg}/large.py, reindex_{word} numbers {word} items from one; the contract says zero-based. "
               "Default the start to zero.",
    "slice": "In src/{pkg}/large.py, keep_{word} mistakenly drops the last value. Return the {word} values unchanged.",
    "join": "In src/{pkg}/large.py, link_{word} appends a stray trailing separator after joining {word} parts. "
            "Join with exactly one separator between parts and none at the ends.",
    "count": "In src/{pkg}/large.py, tally_{word} misses marker letters written in other cases. Count them "
             "case-insensitively.",
    "normalize": "In src/{pkg}/large.py, tidy_{word} leaves runs of inner whitespace alone. Collapse every "
                 "inner whitespace run to a single space and trim the ends.",
}

CHECKS = {
    "strip": (
        "from {pkg_path} import clip_{word}\n"
        "assert clip_{word}('  x  ') == 'x'\n"
        "assert clip_{word}('-_-.x..-_') == 'x'\n"
        "assert clip_{word}('\\t\\n-.x. -\\n') == 'x'\n"
    ),
    "threshold": (
        "from {pkg_path} import flags_{word}\n"
        "assert flags_{word}(10, 10) is True\n"
        "assert flags_{word}(9, 10) is False\n"
        "assert flags_{word}(11, 10) is True\n"
    ),
    "bound": (
        "from {pkg_path} import cap_{word}\n"
        "assert cap_{word}(50, 10, 40) == 40\n"
        "assert cap_{word}(3, 10, 40) == 3\n"
    ),
    "round": (
        "from {pkg_path} import share_{word}\n"
        "assert share_{word}(7, 2) == 4\n"
        "assert share_{word}(6, 2) == 3\n"
        "assert share_{word}(5, 2) == 3\n"
    ),
    "case": (
        "from {pkg_path} import finds_{word}\n"
        "assert finds_{word}('Find The Item', 'the') is True\n"
        "assert finds_{word}('nothing here', 'zzz') is False\n"
    ),
    "default": (
        "from {pkg_path} import reindex_{word}\n"
        "assert reindex_{word}(['a', 'b'])[0] == 'a'\n"
        "assert reindex_{word}(['a'], start=5)[5] == 'a'\n"
    ),
    "slice": (
        "from {pkg_path} import keep_{word}\n"
        "assert keep_{word}([1, 2, 3]) == [1, 2, 3]\n"
        "assert keep_{word}([]) == []\n"
    ),
    "join": (
        "from {pkg_path} import link_{word}\n"
        "assert link_{word}(['a', 'b']) == 'a-b'\n"
        "assert link_{word}(['a']) == 'a'\n"
        "assert link_{word}([], '-') == ''\n"
    ),
    "count": (
        "from {pkg_path} import tally_{word}\n"
        "assert tally_{word}('aAa', 'a') == 3\n"
        "assert tally_{word}('bbb', 'a') == 0\n"
    ),
    "normalize": (
        "from {pkg_path} import tidy_{word}\n"
        "assert tidy_{word}('  a   b  ') == 'a b'\n"
        "assert tidy_{word}('a\\tb\\nc') == 'a b c'\n"
    ),
}


def write_repo(repo: str, spec: dict, source: str, tasks: list[dict]) -> None:
    root = FIXTURES / repo
    pkg = spec["pkg"]
    (root / "src" / pkg).mkdir(parents=True, exist_ok=True)
    (root / "src" / pkg / "__init__.py").write_text("")
    (root / "src" / pkg / "large.py").write_text(source)
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    suite = ["from {0}.large import *".format(pkg), "", "def test_import():", "    assert True", ""]
    (tests / "test_suite.py").write_text("\n".join(suite))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    all_tasks: list[dict] = []
    for repo, spec in REPOS.items():
        source, tasks = build_repo(repo, spec)
        pkg_path = f"{spec['pkg']}.large"
        for t in tasks:
            t["check"] = t["check"].format(pkg_path=pkg_path)
        if args.emit:
            write_repo(repo, spec, source, tasks)
        all_tasks += tasks

    if args.emit:
        public = [{k: v for k, v in t.items() if k not in ("ref_find", "ref_replace")} for t in all_tasks]
        EMIT_PATH.write_text(json.dumps(public, indent=1) + "\n")
        solutions = {t["id"]: [t["ref_find"], t["ref_replace"]] for t in all_tasks}
        (HERE / "holdout_solutions.json").write_text(json.dumps(solutions, indent=1) + "\n")
        sys.path.insert(0, str(HERE.parents[1] / "src"))
        from mintok.funnel import task_fingerprint
        fingerprints = {t["id"]: task_fingerprint(t, FIXTURES / t["repo"]) for t in all_tasks}
        (HERE / "holdout_fingerprints.json").write_text(json.dumps(fingerprints, indent=1) + "\n")
        print(f"emitted {len(all_tasks)} tasks -> {EMIT_PATH.name} (+ holdout_solutions.json, holdout_fingerprints.json)")

    if args.validate:
        failures = 0
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as td:
            for t in all_tasks:
                root = Path(td) / t["id"]
                src = FIXTURES / t["repo"]
                subprocess.run(["cp", "-r", str(src), str(root)], check=True)
                # pristine: check must FAIL
                out = subprocess.run(
                    [sys.executable, "-B", "-c", t["check"].replace("root", repr(str(root)), 1)],
                    capture_output=True, text=True, env=env,
                )
                if out.returncode == 0:
                    print(f"NOT DISCRIMINATING (passes pristine): {t['id']}")
                    failures += 1
                # pristine: suite must PASS
                suite = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", "-q", str(root / "tests")],
                    capture_output=True, text=True,
                    env={**env, "PYTHONPATH": str(root / "src")},
                )
                if suite.returncode != 0:
                    print(f"SUITe RED pristine: {t['id']}")
                    failures += 1
                # reference fix: check must PASS
                large = root / "src" / t["repo"] / "large.py"
                text = large.read_text()
                if t["ref_find"] not in text:
                    print(f"REF FIX NOT APPLICABLE: {t['id']}")
                    failures += 1
                    continue
                # remove any compiled pyc before testing fix
                for pyc in (root / "src").rglob("*.pyc"):
                    pyc.unlink()
                large.write_text(text.replace(t["ref_find"], t["ref_replace"], 1))
                out2 = subprocess.run(
                    [sys.executable, "-B", "-c", t["check"].replace("root", repr(str(root)), 1)],
                    capture_output=True, text=True, env=env,
                )
                if out2.returncode != 0:
                    print(f"REF FIX DOES NOT PASS: {t['id']} {out2.stderr.strip()[-120:]}")
                    failures += 1
                suite2 = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", "-q", str(root / "tests")],
                    capture_output=True, text=True,
                    env={**env, "PYTHONPATH": str(root / "src")},
                )
                if suite2.returncode != 0:
                    print(f"SUITe BROKEN by ref fix: {t['id']}")
                    failures += 1
        print("validation:", "OK" if failures == 0 else f"{failures} problems")
        return 0 if failures == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

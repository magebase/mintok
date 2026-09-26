"""Experiment funnel: cheap screens first, frontier compute last.

A full 120-task run is the final exam. New arms die early and cheaply:

  static integrity checks (no model calls)
    -> FAST-8 smoke suite (8 representative tasks)
    -> DEV-15 (only smoke survivors; sequential stopping)
    -> CONFIRM-30 (only dev survivors)
    -> EVAL-120 (only shippable candidates)

Supporting primitives: deterministic sequential stopping rules, a control
trajectory cache keyed by everything that could change control behavior,
immutable task fingerprints verified before any worker launches, an adaptive
concurrency pool that backs off on provider throttling, and offline metric
recomputation from raw trajectory logs (zero agent runs for new metrics).

Domain logic only: the funnel driver (spawning workers, running checkers)
lives in the benchmark harness and calls into these functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

# ---------------------------------------------------------------------------
# Slicer promotion: the pre-registered bar for the third backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlicerRun:
    """One live slicer trajectory, summarized."""

    task_id: str
    solved: bool
    tokens: int  # tool-context tokens, same convention as run records
    turns: int
    slice_tokens: int  # output tokens delivered by the slice tool
    fallback_tokens: int  # output tokens from raw-source reads
    expanded: bool  # slice needed the expanded budget


@dataclass(frozen=True, slots=True)
class SlicerPromotion:
    verdict: str  # excellent | strong | promote | reject
    solved: int
    attempted: int
    tokens_per_solved: float
    acceptance_rate: float  # solved without the slice being abandoned
    fallback_rate: float  # tasks where raw reads dominated the slice
    expanded_rate: float
    turns_per_attempt: float
    p95_tokens: int
    max_tokens: int
def slicer_promotion(
    runs: Iterable[SlicerRun],
    solve_target: int,
    bar_good: int = 800,
    bar_strong: int = 600,
    bar_excellent: int = 400,
) -> SlicerPromotion:
    """Verdict against the frozen promotion bar.

    Slice acceptance: among *solved* tasks, the share where the slice
    stayed the dominant information channel (raw-source fallback did not
    outspend it). A solved-but-abandoned slice does not credit the
    representation; the fallback rate counts abandoned tasks over all
    attempts.
    """
    runs = list(runs)
    attempted = len(runs)
    solved = sum(r.solved for r in runs)
    solved_rows = [r for r in runs if r.solved]
    accepted = sum(1 for r in solved_rows if r.slice_tokens >= r.fallback_tokens)
    abandoned_solved = sum(1 for r in solved_rows if r.fallback_tokens > r.slice_tokens)
    toks = sorted(r.tokens for r in runs)

    def pct(p: float) -> int:
        return toks[min(len(toks) - 1, math.ceil(p * len(toks)) - 1)]

    tokens_per_solved = sum(r.tokens for r in solved_rows) / solved if solved else float("inf")
    if solved < solve_target or tokens_per_solved > bar_good:
        verdict = "reject"
    elif tokens_per_solved <= bar_excellent:
        verdict = "excellent"
    elif tokens_per_solved <= bar_strong:
        verdict = "strong"
    else:
        verdict = "promote"
    return SlicerPromotion(
        verdict=verdict,
        solved=solved,
        attempted=attempted,
        tokens_per_solved=tokens_per_solved,
        acceptance_rate=accepted / solved if solved else 0.0,
        fallback_rate=abandoned_solved / attempted if attempted else 0.0,
        expanded_rate=sum(r.expanded for r in runs) / attempted if attempted else 0.0,
        turns_per_attempt=sum(r.turns for r in runs) / attempted if attempted else 0.0,
        p95_tokens=pct(0.95) if toks else 0,
        max_tokens=max(toks) if toks else 0,
    )


# ---------------------------------------------------------------------------
# Control-manifest guard: reuse frozen controls only when nothing that could
# change their behavior differs. Matching tasks and token conventions is not
# enough — provider, model, reasoning effort, system prompt, toolset, harness
# version, and task-set hashes all have to agree.
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = (
    "provider",
    "model",
    "reasoning_effort",
    "system_prompt_hash",
    "toolset_hash",
    "harness_version",
    "task_set_hash",
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    provider: str
    model: str
    reasoning_effort: str
    system_prompt_hash: str
    toolset_hash: str
    harness_version: str
    task_set_hash: str

    def to_json(self) -> str:
        return json.dumps({f: getattr(self, f) for f in MANIFEST_FIELDS}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RunManifest":
        payload = json.loads(text)
        return cls(**{f: payload[f] for f in MANIFEST_FIELDS if f in payload})


def manifests_compatible(frozen: RunManifest | None, live: RunManifest) -> tuple[bool, list[str]]:
    """(compatible, reasons). A missing frozen manifest is never compatible."""
    if frozen is None:
        return False, ["frozen control manifest missing; control provenance unknown"]
    reasons = [
        f"{field_}: frozen={getattr(frozen, field_)!r} live={getattr(live, field_)!r}"
        for field_ in MANIFEST_FIELDS
        if getattr(frozen, field_) != getattr(live, field_)
    ]
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Failure attribution: why did a slicer task fail?
# ---------------------------------------------------------------------------

#: decision order encodes the cheapest-to-verify evidence first.
def attribute_failure(
    *,
    solved: bool,
    target_in_slice: bool,
    slice_dominated: bool,
    slice_truncated: bool,
    edit_rejections: int,
    suite_ok: bool,
) -> str:
    """Classify one slicer-run failure from trajectory evidence.

    Classes: ``bad_slice`` (the true working area was never in the package),
    ``insufficient_slice`` (target present but content cut), ``fallback_needed``
    (the agent had to leave the slice to make progress), ``edit_tool_limitation``
    (the line-range patcher rejected the work), ``checker_stochastic`` (the
    suite passed but the checker disagrees — flake or stochastic turn), and
    ``agent_reasoning_failure`` (everything needed was available).
    """
    if solved:
        return ""
    if not target_in_slice:
        return "bad_slice"
    if slice_truncated:
        return "insufficient_slice"
    if not slice_dominated:
        return "fallback_needed"
    if edit_rejections > 0:
        return "edit_tool_limitation"
    if suite_ok:
        return "checker_stochastic"
    return "agent_reasoning_failure"

# ---------------------------------------------------------------------------
# FAST-8: the high-information smoke suite
# ---------------------------------------------------------------------------

#: task id -> behavior class it discriminates. Chosen from trajectory
#: history: each entry either separated arms on the dev set or is a known
#: pathological case. Class labels reuse the eval strata where one applies.
#: Frozen with the eval set so FAST-8 comparisons stay meaningful across arms.
FAST8: tuple[tuple[str, str], ...] = (
    ("shopcart-lookup-03", "simple_lookup"),  # control's genuine miss; read-win class
    ("webledger-cross-01", "cross_file_bug"),  # cross-file modification
    ("pipeline-api-03", "api_signature_propagation"),  # revisit-heavy keyword threading
    ("notesrv-schema-02", "schema_or_framework_change"),  # serialized-shape change + consumers
    ("biglib-large-05", "large_file_navigation"),  # huge-module navigation
    ("webledger-api-02", "simple_api_extension"),  # ordinary parameterized edit
    ("new-pack-is-empty", "semantic_trap"),  # known semantic-reasoning trap
    ("shopcart-feature-05", "feature_addition"),  # output-format sensitive feature
)

FAST8_IDS: frozenset[str] = frozenset(task_id for task_id, _ in FAST8)


def fast8_suite(available: Iterable[str]) -> tuple[str, ...]:
    """The FAST-8 smoke tasks, in fixed order, validated against the pool."""
    available = set(available)
    missing = sorted(FAST8_IDS - available)
    if missing:
        raise KeyError(f"FAST-8 references unknown tasks: {', '.join(missing)}")
    return tuple(task_id for task_id, _ in FAST8)


def fast8_classes() -> dict[str, str]:
    """task id -> behavior class for the smoke suite."""
    return dict(FAST8)


# ---------------------------------------------------------------------------
# Sequential stopping
# ---------------------------------------------------------------------------

#: paired-count checkpoints and the token-ratio kill thresholds. At n>=5 an
#: arm using 1.25x the control tokens per solved task is dominated; at n>=10
#: 1.15x; at n>=15 anything without a meaningful (<0.95x) gain dies. The
#: solve guard applies from 5 pairs on: no arm may buy tokens with success.
DEFAULT_CHECKPOINTS: tuple[tuple[int, float], ...] = ((5, 1.25), (10, 1.15), (15, 0.95))
DEFAULT_SOLVE_GUARD_PP: float = 0.05  # absolute solve-rate drop, as a fraction


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    """One task run under both arms."""

    task_id: str
    control_solved: bool
    arm_solved: bool
    control_tokens: int
    arm_tokens: int


@dataclass(frozen=True, slots=True)
class Verdict:
    """Sequential-stopping decision after the latest pair."""

    action: str  # "continue" | "kill"
    reason: str
    pairs: int
    token_ratio: float  # arm tokens per solved task / control tokens per solved task
    control_solve: float
    arm_solve: float


def _ratio(pairs: list[PairedOutcome]) -> tuple[float, float, float]:
    control_solved = sum(p.control_solved for p in pairs)
    arm_solved = sum(p.arm_solved for p in pairs)
    control_tok = sum(p.control_tokens for p in pairs)
    arm_tok = sum(p.arm_tokens for p in pairs)
    control_rate = control_solved / len(pairs)
    arm_rate = arm_solved / len(pairs)
    control_per_solved = control_tok / control_solved if control_solved else math.inf
    arm_per_solved = arm_tok / arm_solved if arm_solved else math.inf
    if math.isinf(control_per_solved) or math.isinf(arm_per_solved):
        ratio = math.inf if control_per_solved == 0 else float("inf")
        if control_per_solved > 0 and arm_per_solved > 0:
            ratio = math.inf
    else:
        ratio = arm_per_solved / control_per_solved
    return ratio, control_rate, arm_rate


def sequential_verdict(
    pairs: Sequence[PairedOutcome],
    checkpoints: tuple[tuple[int, float], ...] = DEFAULT_CHECKPOINTS,
    solve_guard_pp: float = DEFAULT_SOLVE_GUARD_PP,
) -> Verdict:
    """Deterministic early-stopping decision over paired outcomes.

    Checkpoints are (minimum pair count, ratio threshold): once ``pairs``
    reaches the count, a token ratio above the threshold kills the arm. The
    final checkpoint doubles as the "no meaningful improvement" rule, so it
    fires when the arm is not at least 5% cheaper than control.
    """
    n = len(pairs)
    ratio, control_rate, arm_rate = _ratio(list(pairs))
    if n >= 5 and arm_rate < control_rate - solve_guard_pp:
        return Verdict(
            "kill",
            f"solve rate dropped {100 * (control_rate - arm_rate):.1f} points",
            n,
            ratio,
            control_rate,
            arm_rate,
        )
    for threshold_n, threshold in checkpoints:
        if n >= threshold_n and ratio > threshold:
            if threshold_n == checkpoints[-1][0] and ratio <= 1.0:
                continue  # not worse than control: no kill on the last rung
            reason = (
                "no meaningful improvement"
                if threshold_n == checkpoints[-1][0]
                else f"token ratio {ratio:.2f}x exceeds kill threshold {threshold:.2f}x"
            )
            return Verdict("kill", reason, n, ratio, control_rate, arm_rate)
    return Verdict("continue", f"{n} pairs, ratio {ratio:.2f}x", n, ratio, control_rate, arm_rate)


# ---------------------------------------------------------------------------
# Control trajectory cache
# ---------------------------------------------------------------------------


def control_key(
    task_id: str,
    repo_snapshot_hash: str,
    harness_version: str,
    model: str,
    effort: str,
    toolset: str,
) -> str:
    """Identity of a control run. Any input that could change the
    trajectory changes the key, so cached controls are only reused when
    every influence is bit-identical."""
    payload = json.dumps(
        {
            "task_id": task_id,
            "repo_snapshot_hash": repo_snapshot_hash,
            "harness_version": harness_version,
            "model": model,
            "effort": effort,
            "toolset": toolset,
            "role": "control",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class ControlCache:
    """File-backed store of finished control trajectories.

    A new arm compares against the same cached control run whenever task,
    repo snapshot, harness, model, effort, and control toolset are
    unchanged; control is never rerun merely because another arm appeared.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None  # a corrupt entry is a miss, never a stale hit

    def put(self, key: str, record: Mapping) -> None:
        self._path(key).write_text(json.dumps(dict(record), sort_keys=True))

    def get_or_run(self, key: str, run: Callable[[], Mapping]) -> tuple[Mapping, bool]:
        """Return (record, from_cache). ``run`` executes only on a miss."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        record = dict(run())
        self.put(key, record)
        return record, False


# ---------------------------------------------------------------------------
# Immutable task fingerprints
# ---------------------------------------------------------------------------


class TaskIntegrityError(RuntimeError):
    """Raised when a task drifted from its fingerprint before a run."""


def _content_hash(paths: Iterable[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(base))):
        if path.is_file():
            digest.update(str(path.relative_to(base)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:32]


def task_fingerprint(task: Mapping, fixture_root: Path | None = None) -> dict:
    """Hash every input a worker consumes: prompt, checker, fixtures.

    A fingerprint mismatch means a wave would test something other than
    what was scored before; refuse to run instead.
    """
    fp = {
        "task_id": task["id"],
        "prompt_hash": hashlib.sha256(task["instruction"].encode()).hexdigest()[:32],
        "checker_hash": hashlib.sha256(task["check"].encode()).hexdigest()[:32],
    }
    if fixture_root is not None and Path(fixture_root).is_dir():
        files = [p for p in Path(fixture_root).rglob("*") if "__pycache__" not in p.parts]
        fp["fixture_hash"] = _content_hash(files, Path(fixture_root))
    return fp


def verify_fingerprint(task: Mapping, stored: Mapping, fixture_root: Path | None = None) -> None:
    """Raise TaskIntegrityError if any hashed part drifted."""
    current = task_fingerprint(task, fixture_root)
    for part in sorted(stored):
        if stored[part] != current.get(part):
            raise TaskIntegrityError(
                f"{task['id']}: {part} drifted ({stored[part]!r} -> {current.get(part)!r})"
            )


# ---------------------------------------------------------------------------
# Adaptive concurrency
# ---------------------------------------------------------------------------


class AdaptivePool:
    """Worker-pool sizing that tracks provider tolerance.

    Starts conservative, grows on clean waves, and shrinks sharply the
    moment the provider throttles. Held deliberately narrow (4..12):
    beyond that, rate-limit storms destroy more wall-clock time than the
    parallelism saves.
    """

    def __init__(self, start: int = 8, minimum: int = 4, maximum: int = 12) -> None:
        if not minimum <= start <= maximum:
            raise ValueError("require minimum <= start <= maximum")
        self.minimum = minimum
        self.maximum = maximum
        self._size = start

    @property
    def size(self) -> int:
        return self._size

    def wave_completed(self, throttled: bool) -> None:
        """Adjust after a wave: +2 when clean, -3 on throttling."""
        if throttled:
            self._size = max(self.minimum, self._size - 3)
        else:
            self._size = min(self.maximum, self._size + 2)

    def borrow(self, limit: int) -> int:
        """Workers for the next wave, never exceeding the pool or remaining work."""
        return min(self._size, limit)


# ---------------------------------------------------------------------------
# Funnel phases
# ---------------------------------------------------------------------------

#: phase name -> (suite size, next phase). "dead" is terminal: an arm killed
#: at any checkpoint stops consuming frontier compute.
PHASES: dict[str, tuple[int, str]] = {
    "smoke8": (8, "dev15"),
    "dev15": (15, "confirm30"),
    "confirm30": (30, "eval120"),
    "eval120": (120, "ship"),
}

PHASE_SEQUENCE = tuple(PHASES)


def next_phase(phase: str, killed: bool) -> str:
    """The phase an arm moves to after its current phase."""
    if killed:
        return "dead"
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase!r}")
    return PHASES[phase][1]


def phase_suite(phase: str, available: Iterable[str]) -> tuple[str, ...]:
    """The task ids a phase runs, deterministic and prefix-stable."""
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase!r}")
    pool = sorted(available)
    size = min(PHASES[phase][0], len(pool))
    if phase == "smoke8":
        return fast8_suite(pool)
    # later phases extend the smoke suite first: early tasks accumulate
    # evidence instead of being abandoned after one phase.
    return tuple(pool[:size])

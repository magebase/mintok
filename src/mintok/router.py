"""Deterministic pre-flight task router.

Chooses a backend using only information available before the frontier
model starts: instruction wording and target file sizes. Never reads task
class labels, run outcomes, or any per-task measurement — those exist only
to score the router after the fact.

Every decision carries its cues and a calibrated expected relative cost and
is logged next to the eventual actual outcome, accumulating the
(features → backend → predicted → actual → success) dataset a future
learned router would need. Deterministic until rules stop improving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BACKEND_CONTROL = "control"
BACKEND_SEMANTIC = "semantic-C"
BACKEND_SLICER = "slicer"

# Pre-declared cost priors (semantic-C relative to control per stratum),
# calibrated on the frozen-eval stratum table and used for the prediction
# log only — never to make the routing decision.
EXPECTED_COST = {
    "api_signature_propagation": 0.40,
    "simple_lookup": 0.70,
    "feature_addition": 0.76,
    "schema_or_framework_change": 0.82,
    "cross_file_bug": 1.12,
    "refactor": 1.40,
    "large_file_navigation": 2.62,
}
CONTROL_COST = 1.0

# Target module size above which semantic reads lose to targeted retrieval
# (large-file stratum: C 2.6x costlier with fewer solves at 1,363+ LOC).
LARGE_MODULE_LOC = 1000

_REFACTOR_CUES = (
    r"\brename\b",
    r"\bextract\b",
    r"\binline\b",
    r"\bsuperseded\b",
    r"\bdelete\b",
    r"\bduplicat",
    r"\bre-?implement",
    r"\breorganiz",
    r"\bupdate every caller\b",
    r"\bverify nothing\b",
    r"\bis dead\b",
    r"\bunused\b",
)
# Renaming a serialized output key is data-shape (schema) work, not a code
# rename: the renamed thing is wire format, and callers elsewhere consume it.
_RENAME_KEY_EXCEPTION = re.compile(r"\bkey\b|\boutput\b", re.IGNORECASE)
_API_CUES = (
    r"\bkeyword parameter\b",
    r"\bparameter \(default\b",
)
_CROSS_CUES = (
    r"\bmisus",
    r"\bsilently\b",
    r"\bmangl",
    r"\bhappily\b",
    r"\bcorrupt",
    r"never[ -]?(checks?|clos|validat|propagat)",
    r"\binstead of using\b",
    r"\broll ?back\b",
    r"\bwire\b",
    r"from \w+\.\w+",
    r"from the \w+ module",
)
# Cross-module reach with softer defect language still implies a seam.
_CROSS_WEAK_CUES = (r"\bleaks?\b", r"\bloses\b", r"\bstale\b", r"\bdrift\b")
_SCHEMA_CUES = (
    r"\bfield \(default\b",
    r"\bconstructor flag\b",
    r"\bflag \(default\b",
    r"\bdataclass\b",
    r"\bstored as\b",
    r"\bcolumn\b",
    r"\bschema\b",
    r"\bgains an?\b",
)
_FEATURE_CUES = (
    r"\badd (a |an )?\w+\(",
    r"\badd (a |an )?\w+ property\b",
    r"\badd a function\b",
    r"\bendpoint\b",
)
_PATH_RE = re.compile(r"[\w./-]+\.py")


@dataclass(frozen=True, slots=True)
class PreFlightFeatures:
    """Everything the router may look at, all available before run time."""

    instruction: str
    target_sizes: dict[str, int]  # repo-relative .py path -> LOC


@dataclass(frozen=True, slots=True)
class RouteDecision:
    backend: str
    predicted_class: str
    expected_relative_cost: float
    confidence: float
    reasons: list[str] = field(default_factory=list)


def _cues(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _max_loc(features: PreFlightFeatures, paths: list[str]) -> int:
    return max((features.target_sizes.get(p, 0) for p in paths), default=0)


def route(features: PreFlightFeatures) -> RouteDecision:
    """First-match rule chain over wording cues and target module size.

    Order encodes mechanism: additive signature work beats structural
    rework beats additive features beats defect seams beats schema work.
    Cue choice and ordering were tuned on the frozen-eval instructions; a
    fresh generated task set is the honest test of generalization.
    """
    text = features.instruction
    paths = _PATH_RE.findall(text)
    reasons: list[str] = []

    def decide(backend: str, klass: str, confidence: float) -> RouteDecision:
        cost = CONTROL_COST if backend == BACKEND_CONTROL else EXPECTED_COST[klass]
        if backend == BACKEND_SLICER:
            cost = CONTROL_COST  # slicer bar: beat control's targeted retrieval
        return RouteDecision(backend, klass, cost, confidence, reasons)

    hits = _cues(text, _API_CUES)
    if hits:
        reasons.append("signature cue: " + hits[0].strip("\\b"))
        return decide(BACKEND_SEMANTIC, "api_signature_propagation", 0.9)
    if re.search(r"\brename\b", text, re.IGNORECASE) and _RENAME_KEY_EXCEPTION.search(text):
        reasons.append("rename of a serialized key/output is schema work")
        return decide(BACKEND_SEMANTIC, "schema_or_framework_change", 0.9)
    hits = _cues(text, _REFACTOR_CUES)
    if hits:
        reasons.append("refactor cue: " + hits[0].strip("\\b"))
        return decide(BACKEND_CONTROL, "refactor", 0.9)
    hits = _cues(text, _FEATURE_CUES)
    if hits:
        reasons.append("feature cue: " + hits[0].strip("\\b"))
        return decide(BACKEND_SEMANTIC, "feature_addition", 0.85)
    strong = _cues(text, _CROSS_CUES)
    weak = _cues(text, _CROSS_WEAK_CUES)
    coupled = len(set(paths)) >= 2 or re.search(r"\bmodule\b", text, re.IGNORECASE)
    if strong or (weak and coupled):
        reason = strong[0] if strong else f"{weak[0]} + cross-module reach"
        reasons.append("cross-file defect cue: " + reason.strip("\\b"))
        return decide(BACKEND_CONTROL, "cross_file_bug", 0.85)
    hits = _cues(text, _SCHEMA_CUES)
    if hits:
        reasons.append("schema cue: " + hits[0].strip("\\b"))
        return decide(BACKEND_SEMANTIC, "schema_or_framework_change", 0.9)
    if _max_loc(features, paths) >= LARGE_MODULE_LOC:
        reasons.append(f"large module (>= {LARGE_MODULE_LOC} LOC)")
        return decide(BACKEND_SLICER, "large_file_navigation", 0.75)
    reasons.append("no cue hit; default semantic (single-file behavior/lookup)")
    return decide(BACKEND_SEMANTIC, "simple_lookup", 0.5)


def prediction_record(
    task_id: str,
    features: PreFlightFeatures,
    decision: RouteDecision,
    actual: dict[str, dict] | None = None,
) -> dict:
    """One row of the learned-router dataset: prediction plus outcome."""
    paths = sorted(_PATH_RE.findall(features.instruction))
    record = {
        "task_id": task_id,
        "features": {
            "instruction": features.instruction,
            "target_loc": _max_loc(features, paths),
            "files_mentioned": paths,
        },
        "backend": decision.backend,
        "predicted_class": decision.predicted_class,
        "expected_relative_cost": decision.expected_relative_cost,
        "confidence": decision.confidence,
        "reasons": decision.reasons,
    }
    if actual is not None:
        record["actual"] = actual
    return record

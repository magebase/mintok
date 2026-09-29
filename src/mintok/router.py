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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mintok.repo_profile import RepoProfile

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
    "monorepo_complexity": 1.00,
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


def compute_utility(
    solved: bool,
    tokens: int,
    value: float = 1.0,
    token_lambda: float = 1e-6,
) -> float:
    """Compute task execution utility: U = value * solved - lambda * tokens.

    Balances task completion against context token consumption.
    """
    return (value if solved else 0.0) - (token_lambda * tokens)


@dataclass(frozen=True, slots=True)
class PreFlightFeatures:
    """Everything the router may look at, all available before run time."""

    instruction: str
    target_sizes: dict[str, int]  # repo-relative .py path -> LOC
    repo_profile: RepoProfile | None = None


@dataclass(frozen=True, slots=True)
class PolicyExpectation:
    """Pre-flight expectation of policy performance on a given task state."""

    policy: str
    p_success: float
    expected_tokens: int
    expected_utility: float


def estimate_policy_expectation(
    policy: str,
    features: PreFlightFeatures,
    value: float = 1.0,
    token_lambda: float = 1e-6,
) -> PolicyExpectation:
    """Explicitly estimate P(success | features, policy) and E(tokens | features, policy).

    Derives expected utility: U_hat = value * P(success) - token_lambda * E(tokens).
    """
    text = features.instruction
    paths = _PATH_RE.findall(text)
    max_loc = _max_loc(features, paths)
    is_monorepo = features.repo_profile is not None and (
        features.repo_profile.complexity_score >= 0.50
        or features.repo_profile.recommended_strategy == "virtualized-shell"
    )

    if is_monorepo:
        if policy in (BACKEND_CONTROL, "virtualized-shell"):
            p, tokens = 0.75, 3200
        elif policy == BACKEND_SLICER:
            p, tokens = 0.45, 4500
        else:
            p, tokens = 0.35, 5800
    elif max_loc >= LARGE_MODULE_LOC:
        if policy == BACKEND_SLICER:
            p, tokens = 0.85, 1600
        elif policy in (BACKEND_CONTROL, "virtualized-shell"):
            p, tokens = 0.70, 4200
        else:
            p, tokens = 0.60, 4800
    elif _cues(text, _REFACTOR_CUES) or _cues(text, _CROSS_CUES):
        if policy in (BACKEND_CONTROL, "virtualized-shell"):
            p, tokens = 0.88, 2400
        else:
            p, tokens = 0.65, 3600
    elif _cues(text, _API_CUES) or _cues(text, _FEATURE_CUES) or _cues(text, _SCHEMA_CUES):
        if policy == BACKEND_SEMANTIC:
            p, tokens = 0.95, 950
        elif policy == "virtualized-shell":
            p, tokens = 0.92, 1400
        else:
            p, tokens = 0.88, 2200
    else:
        if policy == BACKEND_SEMANTIC:
            p, tokens = 0.90, 1100
        else:
            p, tokens = 0.85, 2100

    u_hat = (value * p) - (token_lambda * tokens)
    return PolicyExpectation(
        policy=policy,
        p_success=p,
        expected_tokens=tokens,
        expected_utility=u_hat,
    )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    backend: str
    predicted_class: str
    expected_relative_cost: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    expected_utility: float = 0.0
    policy_expectations: dict[str, PolicyExpectation] = field(default_factory=dict)


def _cues(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _max_loc(features: PreFlightFeatures, paths: list[str]) -> int:
    return max((features.target_sizes.get(p, 0) for p in paths), default=0)


def route(features: PreFlightFeatures) -> RouteDecision:
    """First-match rule chain over wording cues and target module size.

    Order encodes mechanism: monorepo complexity beats additive signature work
    beats structural rework beats additive features beats defect seams beats
    schema work.
    """
    text = features.instruction
    paths = _PATH_RE.findall(text)
    reasons: list[str] = []

    exp_control = estimate_policy_expectation(BACKEND_CONTROL, features)
    exp_semantic = estimate_policy_expectation(BACKEND_SEMANTIC, features)
    exp_slicer = estimate_policy_expectation(BACKEND_SLICER, features)
    expectations = {
        BACKEND_CONTROL: exp_control,
        BACKEND_SEMANTIC: exp_semantic,
        BACKEND_SLICER: exp_slicer,
    }

    def decide(backend: str, klass: str, confidence: float) -> RouteDecision:
        cost = CONTROL_COST if backend == BACKEND_CONTROL else EXPECTED_COST[klass]
        if backend == BACKEND_SLICER:
            cost = CONTROL_COST  # slicer bar: beat control's targeted retrieval
        chosen_exp = expectations.get(backend, exp_control)
        return RouteDecision(
            backend=backend,
            predicted_class=klass,
            expected_relative_cost=cost,
            confidence=confidence,
            reasons=reasons,
            expected_utility=chosen_exp.expected_utility,
            policy_expectations=expectations,
        )

    if features.repo_profile is not None and features.repo_profile.recommended_strategy == "virtualized-shell":
        reasons.append(
            f"monorepo complexity ({features.repo_profile.complexity_score:.2f}) prioritizes broad shell execution"
        )
        return decide(BACKEND_CONTROL, "monorepo_complexity", 0.90)

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
            "repo_complexity": features.repo_profile.complexity_score if features.repo_profile else 0.0,
            "recommended_strategy": features.repo_profile.recommended_strategy if features.repo_profile else "compressed-first",
        },
        "backend": decision.backend,
        "predicted_class": decision.predicted_class,
        "expected_relative_cost": decision.expected_relative_cost,
        "confidence": decision.confidence,
        "reasons": decision.reasons,
        "expected_utility": decision.expected_utility,
        "policy_expectations": {
            k: {
                "policy": v.policy,
                "p_success": v.p_success,
                "expected_tokens": v.expected_tokens,
                "expected_utility": v.expected_utility,
            }
            for k, v in decision.policy_expectations.items()
        },
    }
    if actual is not None:
        record["actual"] = actual
        record["utilities"] = {
            backend_name: compute_utility(
                solved=data.get("solved", False),
                tokens=data.get("tokens", 0),
            )
            for backend_name, data in actual.items()
        }
        chosen_backend = decision.backend
        chosen_exp = decision.policy_expectations.get(chosen_backend)
        chosen_actual = actual.get(chosen_backend, {})
        record["predicted_psolve"] = chosen_exp.p_success if chosen_exp else 0.5
        record["predicted_tokens"] = chosen_exp.expected_tokens if chosen_exp else 2000
        record["predicted_utility"] = chosen_exp.expected_utility if chosen_exp else 0.0
        record["chosen_policy"] = chosen_backend
        record["actual_solved"] = chosen_actual.get("solved", False)
        record["actual_tokens"] = chosen_actual.get("tokens", 0)
        record["actual_utility"] = record["utilities"].get(chosen_backend, 0.0)
        max_u = max(record["utilities"].values()) if record["utilities"] else 0.0
        record["utility_regret"] = max(0.0, max_u - record["actual_utility"])
        best_backend = max(record["utilities"], key=record["utilities"].get) if record["utilities"] else chosen_backend
        record["routing_regret"] = (best_backend != chosen_backend)
    return record


def evaluate_router_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute calibration error, prediction MAE, and routing regret across task records."""
    if not records:
        return {
            "tasks": 0,
            "psolve_calibration_mae": 0.0,
            "brier_score": 0.0,
            "token_mae": 0.0,
            "mean_utility_regret": 0.0,
            "routing_regret_rate": 0.0,
        }

    psolve_errors = []
    brier_scores = []
    token_errors = []
    utility_regrets = []
    routing_regrets = []

    for r in records:
        p_pred = r.get("predicted_psolve", 0.5)
        t_pred = r.get("predicted_tokens", 2000)

        actual_solved = 1.0 if r.get("actual_solved", False) else 0.0
        actual_tokens = r.get("actual_tokens", 0)

        psolve_errors.append(abs(p_pred - actual_solved))
        brier_scores.append((p_pred - actual_solved) ** 2)
        if actual_tokens > 0:
            token_errors.append(abs(t_pred - actual_tokens))

        if "utility_regret" in r:
            utility_regrets.append(r["utility_regret"])
        if "routing_regret" in r:
            routing_regrets.append(1.0 if r["routing_regret"] else 0.0)

    n = len(records)
    return {
        "tasks": n,
        "psolve_calibration_mae": sum(psolve_errors) / n if psolve_errors else 0.0,
        "brier_score": sum(brier_scores) / n if brier_scores else 0.0,
        "token_mae": sum(token_errors) / len(token_errors) if token_errors else 0.0,
        "mean_utility_regret": sum(utility_regrets) / len(utility_regrets) if utility_regrets else 0.0,
        "routing_regret_rate": sum(routing_regrets) / len(routing_regrets) if routing_regrets else 0.0,
    }

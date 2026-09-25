"""Inference profiler: quantify avoidable spend in agent session records.

The profiler optimizes nothing. It makes waste visible — the free first stage
of the open-core funnel. Its formulas are published and intentionally simple
heuristics; categories may overlap, so they are never multiplied together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mintok.records import SessionRecord

DETERMINISTIC_TASK_CLASSES = frozenset({"deterministic"})
POOR_CACHE_THRESHOLD = 0.5
POOR_CACHE_SHARE = 0.25

FORMULAS = {
    "repeated context": "price every read beyond the first at the session's average cost per token",
    "overpowered model": "flags the full cost of frontier sessions on deterministic task classes",
    "poor cache use": "flags 25% of cost when fewer than half the input tokens hit cache",
}
CATEGORY_ORDER = ("repeated context", "overpowered model", "poor cache use")


@dataclass(frozen=True, slots=True)
class Profile:
    sessions: int
    total: float
    categories: dict[str, float] = field(default_factory=dict)

    @property
    def raw_categories_total(self) -> float:
        return sum(self.categories.values())

    @property
    def categories_overlap(self) -> bool:
        """True when the same dollar was flagged by more than one heuristic."""
        return self.raw_categories_total > self.total

    @property
    def avoidable(self) -> float:
        """Upper bound on recoverable spend, capped at total spend."""
        return min(self.raw_categories_total, self.total)

    @property
    def improvement_multiple(self) -> float:
        remaining = self.total - self.avoidable
        return self.total / remaining if remaining > 0 else float("inf")


def profile_sessions(sessions: Iterable[SessionRecord]) -> Profile:
    sessions = list(sessions)

    repeated = 0.0
    for session in sessions:
        extra_tokens = sum((read.count - 1) * read.tokens for read in session.context_reads)
        repeated += extra_tokens * session.cost_per_token

    overpowered = sum(s.usd for s in sessions if s.task_class in DETERMINISTIC_TASK_CLASSES)
    poor_cache = sum(s.usd * POOR_CACHE_SHARE for s in sessions if s.cache_read_ratio < POOR_CACHE_THRESHOLD)

    return Profile(
        sessions=len(sessions),
        total=sum(s.usd for s in sessions),
        categories={
            "repeated context": repeated,
            "overpowered model": overpowered,
            "poor cache use": poor_cache,
        },
    )


def render_profile(profile: Profile) -> str:
    lines = [f"Inference profile ({profile.sessions} sessions)"]
    lines.append(f"Total spend              ${profile.total:,.2f}")
    for category in CATEGORY_ORDER:
        lines.append(f"  {category:<23}${profile.categories.get(category, 0.0):,.2f}")
    lines.append(f"Estimated avoidable      ${profile.avoidable:,.2f}")
    multiple = profile.improvement_multiple
    lines.append(
        "Potential improvement    n/a (all spend flagged)"
        if multiple == float("inf")
        else f"Potential improvement    {multiple:.2f}x"
    )
    if profile.categories_overlap:
        lines.append("note: waste categories may overlap; avoidable is capped at total spend")
    return "\n".join(lines)

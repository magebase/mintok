"""Observation Dependency Graph and Surgical TTL Invalidation for MinTok.

Maps observations -> symbols -> files -> patches with granular lifecycle management:
1. Tiered TTL Modes:
   - RUN: Valid for entire agent session (e.g. repo architecture, environment info).
   - UNTIL_PATCH: Invalidated by any code edit (e.g. test execution results, stack traces).
   - UNTIL_CODE_CHANGE: Surgically invalidated only when patch intersects specific file line spans or symbols.
   - IMMEDIATE: Single-turn lifespan (transient command outputs).
2. Surgical Invalidation:
   - When a patch modifies file.py:L10-25, observations covering file.py:L100-150 are preserved,
     saving thousands of context tokens that naive whole-file cache invalidation wastes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class TTLMode(str, Enum):
    """Lifecycle policy governing observation validity."""

    RUN = "RUN"
    UNTIL_PATCH = "UNTIL_PATCH"
    UNTIL_CODE_CHANGE = "UNTIL_CODE_CHANGE"
    IMMEDIATE = "IMMEDIATE"


@dataclass(frozen=True, slots=True)
class SymbolSpan:
    """Exact source range covered by an observation."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""

    def overlaps_with(self, start: int, end: int) -> bool:
        """Check if this span overlaps with given line range [start, end]."""
        return max(self.start_line, start) <= min(self.end_line, end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CachedObservation:
    """A cached tool observation with dependency tracking metadata."""

    observation_id: str
    observation_type: str
    content_summary: str
    files_covered: list[str]
    spans: list[SymbolSpan]
    symbols: list[str]
    ttl_mode: TTLMode
    created_at_turn: int
    token_size: int
    is_valid: bool = True
    invalidated_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "content_summary": self.content_summary,
            "files_covered": list(self.files_covered),
            "spans": [s.to_dict() for s in self.spans],
            "symbols": list(self.symbols),
            "ttl_mode": self.ttl_mode.value,
            "created_at_turn": self.created_at_turn,
            "token_size": self.token_size,
            "is_valid": self.is_valid,
            "invalidated_reason": self.invalidated_reason,
        }


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    """Outcome of an observation cache invalidation pass."""

    invalidated_ids: list[str]
    retained_ids: list[str]
    retained_tokens: int
    invalidation_reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidated_ids": list(self.invalidated_ids),
            "retained_ids": list(self.retained_ids),
            "retained_tokens": self.retained_tokens,
            "invalidation_reasons": dict(self.invalidation_reasons),
        }


class ObservationDependencyGraph:
    """Tracks observation dependencies and applies surgical invalidation."""

    def __init__(self) -> None:
        self.observations: dict[str, CachedObservation] = {}

    def register_observation(
        self,
        observation_id: str,
        observation_type: str,
        content_summary: str,
        files_covered: Sequence[str],
        spans: Sequence[SymbolSpan] = (),
        symbols: Sequence[str] = (),
        ttl_mode: TTLMode = TTLMode.UNTIL_CODE_CHANGE,
        turn: int = 1,
        token_size: int = 500,
    ) -> CachedObservation:
        """Register a new observation with source spans and TTL policy."""
        obs = CachedObservation(
            observation_id=observation_id,
            observation_type=observation_type,
            content_summary=content_summary,
            files_covered=list(files_covered),
            spans=list(spans),
            symbols=list(symbols),
            ttl_mode=ttl_mode,
            created_at_turn=turn,
            token_size=token_size,
            is_valid=True,
        )
        self.observations[observation_id] = obs
        return obs

    def invalidate_on_patch(
        self,
        file_path: str,
        patch_start_line: int,
        patch_end_line: int,
        patch_symbols: Sequence[str] = (),
    ) -> InvalidationResult:
        """Surgically invalidate only observations affected by a patch hunk."""
        invalidated_ids = []
        retained_ids = []
        invalidation_reasons = {}

        patch_sym_set = set(patch_symbols)

        for obs_id, obs in self.observations.items():
            if not obs.is_valid:
                continue

            should_invalidate = False
            reason = ""

            if obs.ttl_mode == TTLMode.UNTIL_PATCH:
                should_invalidate = True
                reason = "Invalidated by code patch (UNTIL_PATCH policy)"
            elif obs.ttl_mode == TTLMode.UNTIL_CODE_CHANGE:
                if file_path in obs.files_covered:
                    # Check span overlap
                    matched_span = False
                    for span in obs.spans:
                        if span.file_path == file_path and span.overlaps_with(patch_start_line, patch_end_line):
                            matched_span = True
                            reason = (
                                f"Surgically invalidated: patch {file_path}:{patch_start_line}-{patch_end_line} "
                                f"overlaps with span {span.start_line}-{span.end_line}"
                            )
                            break

                    # Check symbol overlap
                    matched_sym = False
                    if not matched_span and patch_sym_set:
                        common = set(obs.symbols).intersection(patch_sym_set)
                        if common:
                            matched_sym = True
                            reason = f"Surgically invalidated: modified symbol(s) {common}"

                    # If no spans specified for this file, conservative file invalidation
                    if not obs.spans and not obs.symbols:
                        matched_span = True
                        reason = f"Invalidated by modification to covered file {file_path}"

                    if matched_span or matched_sym:
                        should_invalidate = True

            if should_invalidate:
                obs.is_valid = False
                obs.invalidated_reason = reason
                invalidated_ids.append(obs_id)
                invalidation_reasons[obs_id] = reason
            else:
                retained_ids.append(obs_id)

        retained_tokens = sum(self.observations[oid].token_size for oid in retained_ids)

        return InvalidationResult(
            invalidated_ids=invalidated_ids,
            retained_ids=retained_ids,
            retained_tokens=retained_tokens,
            invalidation_reasons=invalidation_reasons,
        )

    def advance_turn(self, new_turn: int) -> list[str]:
        """Invalidate IMMEDIATE observations from previous turns."""
        invalidated = []
        for obs_id, obs in self.observations.items():
            if obs.is_valid and obs.ttl_mode == TTLMode.IMMEDIATE and obs.created_at_turn < new_turn:
                obs.is_valid = False
                obs.invalidated_reason = f"Expired IMMEDIATE TTL at turn {new_turn}"
                invalidated.append(obs_id)
        return invalidated

    def get_valid_observations(self) -> list[CachedObservation]:
        """Return all currently valid observations."""
        return [obs for obs in self.observations.values() if obs.is_valid]

    def get_valid_token_count(self) -> int:
        """Calculate total context tokens occupied by valid observations."""
        return sum(obs.token_size for obs in self.observations.values() if obs.is_valid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_observations": len(self.observations),
            "valid_observations": len(self.get_valid_observations()),
            "valid_tokens": self.get_valid_token_count(),
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
        }

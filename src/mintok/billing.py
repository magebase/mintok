"""API billing instrumentation.

Turns raw token usage into itemized, verifiable dollar costs against a
price table. Pure logic: no network, no model-specific secrets; the only
I/O is reading an optional JSON override file.

Token convention: ``input_tokens`` EXCLUDES cached tokens. Cache hits are
counted in ``cached_input_tokens`` and cache writes in
``cache_write_tokens``. This matches Anthropic's usage reporting, where
``input_tokens`` already excludes cache reads and cache creation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageRecord:
    """Token usage for one model call.

    ``input_tokens`` EXCLUDES cached tokens (see module docstring): cache
    hits belong in ``cached_input_tokens``, cache writes in
    ``cache_write_tokens``.
    """

    model: str
    input_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name == "model":
                continue
            value = getattr(self, f.name)
            if value < 0:
                raise ValueError(f"{f.name} must be nonnegative, got {value}")


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Published USD rates per 1M tokens for one model."""

    input: float
    cached_input: float
    cache_write: float
    output: float
    reasoning: float


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Itemized USD cost for one usage record, plus the total."""

    model: str
    items: dict[str, float] = field(default_factory=dict)
    total: float = 0.0


class PriceTable:
    """Model price lookup with optional JSON overrides over the defaults."""

    # List-price snapshots (USD per 1M tokens) from the public pricing
    # pages, captured 2026-06. List prices drift; use load() overrides
    # rather than editing these constants.
    DEFAULTS: ClassVar[dict[str, dict[str, float]]] = {
        "claude-sonnet-4-5": {
            "input": 3.0,
            "cached_input": 0.30,
            "cache_write": 3.75,
            "output": 15.0,
            "reasoning": 0.0,
        },
        "gpt-5": {
            "input": 1.25,
            "cached_input": 0.125,
            "cache_write": 0.0,
            "output": 10.0,
            "reasoning": 10.0,
        },
    }

    def __init__(self, overrides: dict[str, dict[str, float]] | None = None) -> None:
        merged = {model: dict(rates) for model, rates in self.DEFAULTS.items()}
        for model, rates in (overrides or {}).items():
            merged.setdefault(model, {}).update(rates)
        self._prices = {model: ModelPrice(**rates) for model, rates in merged.items()}

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        """Load the default table, merged with optional JSON overrides.

        The override file maps ``{model: {rate: value}}``; unknown models
        are added, known models get their listed rates overridden.
        """
        if path is None:
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(overrides=payload)

    @property
    def models(self) -> list[str]:
        return sorted(self._prices)

    def cost(self, model: str, usage: UsageRecord) -> CostBreakdown:
        """Price one usage record, itemized by rate category."""
        try:
            price = self._prices[model]
        except KeyError:
            known = ", ".join(sorted(self._prices))
            raise KeyError(f"unknown model {model!r}; known models: {known}") from None
        items = {
            "input": usage.input_tokens * price.input / 1_000_000,
            "cached_input": usage.cached_input_tokens * price.cached_input / 1_000_000,
            "cache_write": usage.cache_write_tokens * price.cache_write / 1_000_000,
            "output": usage.output_tokens * price.output / 1_000_000,
            "reasoning": usage.reasoning_tokens * price.reasoning / 1_000_000,
        }
        return CostBreakdown(model=model, items=items, total=sum(items.values()))


def billing_row(
    run_id: str,
    task_id: str,
    turn: int,
    usage: UsageRecord,
    breakdown: CostBreakdown,
) -> dict:
    """Flatten one billed call into a JSONL-ready row."""
    return {
        "run_id": run_id,
        "task_id": task_id,
        "turn": turn,
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        **{f"cost_{name}_usd": amount for name, amount in breakdown.items.items()},
        "cost_total_usd": breakdown.total,
    }

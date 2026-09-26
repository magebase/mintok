"""Deterministic retry policy (no sleeping, no wall clock)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pipeline.errors import ExhaustedRetries


@dataclass
class RetryPolicy:
    """Retry a callable up to ``max_attempts`` times when it raises one of
    ``retry_on``. Attempts are counted, never slept on."""

    max_attempts: int = 3
    retry_on: tuple = (Exception,)
    on_give_up: str = "raise"  # or "return_none"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        """True when ``attempt`` (0-based, just failed) may try again."""
        if not isinstance(exc, self.retry_on):
            return False
        return attempt + 1 < self.max_attempts


def attempt(fn: Callable[[], object], policy: RetryPolicy) -> object:
    """Call fn() retrying per policy; raise ExhaustedRetries when giving up."""
    last: BaseException | None = None
    for attempt_no in range(policy.max_attempts):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - policy filters below
            last = exc
            if not policy.should_retry(attempt_no, exc):
                break
    if policy.on_give_up == "return_none":
        return None
    raise ExhaustedRetries(f"gave up after {policy.max_attempts} attempts: {last}") from last

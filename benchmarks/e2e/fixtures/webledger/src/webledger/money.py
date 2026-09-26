"""Money helpers. The ledger records integer cents everywhere."""

from __future__ import annotations


def to_cents(amount: float) -> int:
    """Convert a decimal amount to cents, half-up on positive values."""
    import math

    return math.floor(amount * 100 + 0.5)


def from_cents(cents: int) -> float:
    """Convert cents back to a decimal amount."""
    return cents / 100.0


def format_amount(cents: int) -> str:
    """Render cents with thousands separators: 1234567 -> "12,345.67"."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    dollars = cents // 100
    return f"{sign}{dollars:,}.{cents % 100:02d}"


def is_zero(cents: int) -> bool:
    return cents == 0

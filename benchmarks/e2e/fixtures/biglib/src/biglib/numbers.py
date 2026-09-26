"""Number helpers, including digit-sum utilities used by checksum paths."""

from __future__ import annotations


def clamp(value: int, low: int, high: int) -> int:
    """Clamp value into the closed interval [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    return max(low, min(high, value))


def digit_sum(value: int) -> int:
    """Sum of decimal digits (absolute value)."""
    return sum(int(d) for d in str(abs(value)))


def luhn_like_checksum(number: int) -> int:
    """Toy checksum: double every other digit and reduce to one digit."""
    digits = [int(d) for d in str(abs(number))]
    doubled = []
    for i, d in enumerate(digits):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        doubled.append(d)
    return digit_sum(sum(doubled)) % 10


def distribute(total: int, parts: int) -> list[int]:
    """Split a total into near-equal integer parts (remainder on the left)."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]

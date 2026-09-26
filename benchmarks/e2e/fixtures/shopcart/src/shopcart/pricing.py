"""Pricing rules: subtotals, discounts and tax."""

from __future__ import annotations

from shopcart.currency import round_half_up
from shopcart.models import Cart

TAX_RATE = 0.0825


def compute_subtotal(cart: Cart) -> int:
    """Sum of all line totals, in cents."""
    return sum(item.line_total_cents for item in cart.items)


def apply_percentage_discount(subtotal_cents: int, percent: float) -> int:
    """Return the subtotal after removing ``percent`` percent, half-up rounded."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be within 0..100")
    discount = round_half_up(subtotal_cents * percent / 100.0)
    return subtotal_cents - discount


def apply_flat_discount(subtotal_cents: int, cents_off: int) -> int:
    """Return the subtotal after a flat deduction, floored at zero."""
    if cents_off < 0:
        raise ValueError("flat discount cannot be negative")
    return max(0, subtotal_cents - cents_off)


def compute_tax(amount_cents: int) -> int:
    """Sales tax on a discounted amount, half-up rounded (duplicated rounding
    logic with shopcart.currency.round_half_up by design)."""
    import math

    return math.floor(amount_cents * TAX_RATE + 0.5)


def compute_total(
    subtotal_cents: int,
    discount_cents: int = 0,
    shipping_cents: int = 0,
) -> dict[str, int]:
    """Build the full price breakdown for a checkout."""
    discounted = max(0, subtotal_cents - discount_cents)
    tax = compute_tax(discounted)
    total = discounted + tax + shipping_cents
    return {
        "subtotal": subtotal_cents,
        "discount": discount_cents,
        "shipping": shipping_cents,
        "tax": tax,
        "total": total,
    }

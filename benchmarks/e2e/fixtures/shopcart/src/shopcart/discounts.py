"""Discount code registry and application rules."""

from __future__ import annotations

from dataclasses import dataclass

from shopcart.pricing import apply_flat_discount, apply_percentage_discount


@dataclass
class DiscountCode:
    """A promotional code. ``kind`` is either "percent" or "flat"."""

    code: str
    kind: str
    value: float
    min_subtotal_cents: int = 0


DISCOUNT_CODES: dict[str, DiscountCode] = {
    "SAVE10": DiscountCode("SAVE10", "percent", 10.0),
    "FIVEOFF": DiscountCode("FIVEOFF", "flat", 500),
    "BIGDEAL": DiscountCode("BIGDEAL", "percent", 25.0, min_subtotal_cents=10000),
    "VIPHALF": DiscountCode("VIPHALF", "percent", 50.0, min_subtotal_cents=25000),
}


def lookup_code(code: str) -> DiscountCode | None:
    """Case-insensitive lookup of a discount code."""
    return DISCOUNT_CODES.get(code.strip().upper())


def validate_code(discount: DiscountCode, subtotal_cents: int) -> tuple[bool, str]:
    """Check a code against the current subtotal; return (ok, reason)."""
    if subtotal_cents < discount.min_subtotal_cents:
        minimum = discount.min_subtotal_cents
        return False, f"requires minimum subtotal of {minimum} cents"
    return True, "ok"


def apply_code(subtotal_cents: int, code: str) -> tuple[int, str]:
    """Apply a discount code to a subtotal.

    Returns ``(new_subtotal, message)``. Unknown or invalid codes leave the
    subtotal unchanged and explain why in the message.
    """
    discount = lookup_code(code)
    if discount is None:
        return subtotal_cents, "unknown code"
    ok, reason = validate_code(discount, subtotal_cents)
    if not ok:
        return subtotal_cents, f"code not applicable: {reason}"
    if discount.kind == "percent":
        new_subtotal = apply_percentage_discount(subtotal_cents, discount.value)
    elif discount.kind == "flat":
        new_subtotal = apply_flat_discount(subtotal_cents, int(discount.value))
    else:
        return subtotal_cents, f"unknown discount kind: {discount.kind}"
    return new_subtotal, f"applied {discount.code}"


def legacy_percent_off(subtotal_cents: int, percent: float) -> int:
    """Superseded by shopcart.pricing.apply_percentage_discount; kept because
    an old batch importer still references it in documentation."""
    if percent > 100:
        percent = 100
    return int(subtotal_cents * (100 - percent) / 100)

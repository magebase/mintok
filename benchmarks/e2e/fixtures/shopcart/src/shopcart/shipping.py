"""Shipping cost estimation by region and weight."""

from __future__ import annotations

from shopcart.models import Cart
from shopcart.pricing import compute_subtotal

REGION_RATES: dict[str, int] = {
    "US": 500,
    "EU": 1200,
    "APAC": 1500,
    "ROW": 2000,
}

FREE_SHIPPING_THRESHOLD_CENTS = 7500
SURCHARGE_PER_KG_CENTS = 100


def estimate_shipping(weight_grams: int, region: str) -> int:
    """Base regional rate covering the first kilogram, plus a surcharge per
    additional kilogram started."""
    try:
        base = REGION_RATES[region]
    except KeyError:
        raise ValueError(f"unknown region: {region}") from None
    if weight_grams <= 1000:
        return base
    extra_kg = (weight_grams - 1000) // 1000
    if (weight_grams - 1000) % 1000:
        extra_kg += 1
    return base + extra_kg * SURCHARGE_PER_KG_CENTS


def shipping_for_cart(cart: Cart, region: str) -> int:
    """Free above the threshold, otherwise the weight estimate."""
    if compute_subtotal(cart) >= FREE_SHIPPING_THRESHOLD_CENTS:
        return 0
    weight = sum(i.product.weight_grams * i.quantity for i in cart.items)
    return estimate_shipping(weight, region)

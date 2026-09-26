import pytest

from shopcart.currency import format_money, parse_money, round_half_up
from shopcart.pricing import (
    apply_flat_discount,
    apply_percentage_discount,
    compute_subtotal,
    compute_tax,
    compute_total,
)

from catalog import BOOK, MUG, NOTEBOOK, TSHIRT


def test_format_money_renders_cents():
    assert format_money(1234) == "$12.34"
    assert format_money(5) == "$0.05"
    assert format_money(-250) == "-$2.50"


def test_parse_money_round_trip():
    assert parse_money("$12.34") == 1234
    assert parse_money("0.05") == 5
    assert parse_money("1,299.00") == 129900
    with pytest.raises(ValueError):
        parse_money("abc")


def test_round_half_up():
    assert round_half_up(2.5) == 3
    assert round_half_up(2.4) == 2


def test_subtotal_sums_line_totals():
    from shopcart.models import Cart

    cart = Cart(cart_id="c1")
    cart.add(BOOK, 2)
    cart.add(MUG, 1)
    assert compute_subtotal(cart) == BOOK.unit_price_cents * 2 + MUG.unit_price_cents


def test_percentage_discount_bounds():
    assert apply_percentage_discount(1000, 25) == 750
    with pytest.raises(ValueError):
        apply_percentage_discount(1000, 150)


def test_flat_discount_floors_at_zero():
    assert apply_flat_discount(1000, 500) == 500
    assert apply_flat_discount(300, 500) == 0


def test_tax_and_total_breakdown():
    tax = compute_tax(1000)
    assert tax == 83  # 8.25% of 1000, half-up
    breakdown = compute_total(1000, discount_cents=100, shipping_cents=500)
    assert breakdown["subtotal"] == 1000
    assert breakdown["tax"] == compute_tax(900)
    assert breakdown["total"] == 900 + breakdown["tax"] + 500


def test_empty_cart_subtotal_is_zero():
    from shopcart.models import Cart

    assert compute_subtotal(Cart(cart_id="empty")) == 0

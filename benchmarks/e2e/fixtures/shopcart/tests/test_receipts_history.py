from shopcart.history import PurchaseHistory
from shopcart.models import Order
from shopcart.receipts import render_receipt


def make_order(n=1, total=2500):
    return Order(
        order_id=f"ORD-{n:04d}",
        lines=[("BOOK-001", "Intro to Testing", 1, 1995), ("MUG-100", "Lab Mug", 1, 825)],
        total_cents=total,
    )


def test_history_records_and_queries():
    history = PurchaseHistory()
    history.record(make_order(1))
    history.record(make_order(2))
    assert len(history) == 2
    assert history.total_spent() == 5000
    assert history.units_sold("MUG-100") == 2
    assert len(history.orders_for("BOOK-001")) == 2
    assert history.orders_for("TEE-L") == []


def test_top_product():
    history = PurchaseHistory()
    history.record(make_order(1))
    order2 = Order("ORD-0002", [("MUG-100", "Lab Mug", 3, 825)], 2475)
    history.record(order2)
    assert history.top_product() == "MUG-100"


def test_top_product_empty_history():
    assert PurchaseHistory().top_product() is None


def test_receipt_renders_lines():
    receipt = render_receipt(make_order(7))
    assert receipt.startswith("Receipt ORD-0007")
    assert "Intro to Testing" in receipt
    assert "x1" in receipt
    assert "$19.95" in receipt
    assert "TOTAL" in receipt

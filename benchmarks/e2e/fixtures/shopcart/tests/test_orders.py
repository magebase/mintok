import pytest

from shopcart.carts import CartError, CartManager
from shopcart.inventory import Inventory, OutOfStockError
from shopcart.orders import OrderService, generate_order_id
from shopcart.shipping import REGION_RATES, estimate_shipping, shipping_for_cart

from catalog import BOOK, MUG, NOTEBOOK, TSHIRT


def test_cart_manager_lifecycle():
    mgr = CartManager()
    cart = mgr.create()
    mgr.add_item(cart.cart_id, BOOK, 2)
    mgr.add_item(cart.cart_id, MUG)
    assert mgr.get(cart.cart_id) is cart
    assert cart.total_units == 3
    assert mgr.remove_item(cart.cart_id, "BOOK-001")
    assert cart.total_units == 1


def test_merge_carts():
    mgr = CartManager()
    a = mgr.create()
    b = mgr.create()
    mgr.add_item(a.cart_id, BOOK)
    mgr.add_item(b.cart_id, MUG, 2)
    merged = mgr.merge(a.cart_id, b.cart_id)
    assert merged.total_units == 3
    assert mgr.active_carts == 1
    with pytest.raises(CartError):
        mgr.get(a.cart_id)


def test_merge_self_raises():
    mgr = CartManager()
    a = mgr.create()
    with pytest.raises(CartError):
        mgr.merge(a.cart_id, a.cart_id)


def test_place_order_reserves_stock():
    service = OrderService(Inventory())
    from shopcart.models import Cart

    cart = Cart(cart_id="c1")
    cart.add(BOOK, 3)
    order = service.place_order(cart, "US")
    assert order.status.value == "placed"
    assert service.inventory.available("BOOK-001") == 22


def test_place_order_out_of_stock():
    service = OrderService(Inventory({"TEE-L": 1}))
    from shopcart.models import Cart

    cart = Cart(cart_id="c2")
    cart.add(TSHIRT, 5)
    with pytest.raises(OutOfStockError):
        service.place_order(cart, "US")


def test_cancel_last_releases_stock():
    service = OrderService(Inventory())
    from shopcart.models import Cart

    cart = Cart(cart_id="c3")
    cart.add(BOOK, 2)
    order = service.place_order(cart, "US")
    assert service.inventory.available("BOOK-001") == 23
    cancelled = service.cancel_last()
    assert cancelled is order
    assert service.inventory.available("BOOK-001") == 25


def test_order_ids_are_sequential():
    first = generate_order_id()
    second = generate_order_id()
    assert second == f"ORD-{int(first.split('-')[1]) + 1:04d}"


def test_shipping_estimate_by_weight():
    assert estimate_shipping(500, "US") == REGION_RATES["US"]  # first kg covered
    # 2.5 kg -> base covers first kg, 2 extra kg surcharges
    assert estimate_shipping(2500, "US") == REGION_RATES["US"] + 200
    with pytest.raises(ValueError):
        estimate_shipping(100, "MARS")


def test_free_shipping_threshold():
    from shopcart.models import Cart

    cart = Cart(cart_id="c4")
    cart.add(TSHIRT, 6)  # 9000 cents subtotal, above threshold
    assert shipping_for_cart(cart, "EU") == 0

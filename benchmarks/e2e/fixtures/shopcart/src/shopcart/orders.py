"""Order placement: cart -> inventory reservation -> priced order."""

from __future__ import annotations

from shopcart.discounts import apply_code
from shopcart.inventory import Inventory, OutOfStockError
from shopcart.models import Cart, Order, OrderStatus
from shopcart.pricing import compute_tax, compute_total
from shopcart.shipping import shipping_for_cart

_ORDER_SEQ = 0


def generate_order_id() -> str:
    """Deterministic sequential order ids: ORD-0001, ORD-0002, ..."""
    global _ORDER_SEQ
    _ORDER_SEQ += 1
    return f"ORD-{_ORDER_SEQ:04d}"


class OrderService:
    """Places orders against an inventory, applying discounts and shipping."""

    def __init__(self, inventory: Inventory) -> None:
        self.inventory = inventory
        self.placed: list[Order] = []

    def place_order(self, cart: Cart, region: str, code: str | None = None) -> Order:
        """Reserve stock and price the cart; raises OutOfStockError on shortage."""
        for item in cart.items:
            if not self.inventory.reserve(item.product.sku, item.quantity):
                raise OutOfStockError(f"insufficient stock for {item.product.sku}")

        subtotal = sum(item.line_total_cents for item in cart.items)
        discount = 0
        message = ""
        if code:
            discounted, message = apply_code(subtotal, code)
            discount = subtotal - discounted
            subtotal = discounted
        shipping = shipping_for_cart(cart, region)
        tax = compute_tax(subtotal)
        total = subtotal + tax + shipping

        order = Order(
            order_id=generate_order_id(),
            lines=[
                (i.product.sku, i.product.name, i.quantity, i.product.unit_price_cents)
                for i in cart.items
            ],
            total_cents=total,
            status=OrderStatus.PLACED,
        )
        self.placed.append(order)
        return order

    def cancel_last(self) -> Order | None:
        """Cancel the most recent order and release its reserved stock."""
        if not self.placed:
            return None
        order = self.placed.pop()
        order.status = OrderStatus.CANCELLED
        for sku, _name, qty, _price in order.lines:
            self.inventory.release(sku, qty)
        return order

    def order_summary(self, order: Order) -> dict[str, int]:
        """Expose the pricing breakdown stored alongside an order."""
        breakdown = compute_total(order.total_cents)
        breakdown["lines"] = order.line_count
        return breakdown

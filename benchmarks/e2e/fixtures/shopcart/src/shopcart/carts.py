"""Cart lifecycle management for a storefront session."""

from __future__ import annotations

from shopcart.models import Cart, Product


class CartError(Exception):
    """Raised when a cart operation cannot be completed."""


class CartManager:
    """Owns the set of active carts keyed by cart id."""

    def __init__(self) -> None:
        self._carts: dict[str, Cart] = {}
        self._next_id = 1

    def create(self) -> Cart:
        cart_id = f"cart-{self._next_id:04d}"
        self._next_id += 1
        cart = Cart(cart_id=cart_id)
        self._carts[cart_id] = cart
        return cart

    def get(self, cart_id: str) -> Cart:
        cart = self._carts.get(cart_id)
        if cart is None:
            raise CartError(f"no such cart: {cart_id}")
        return cart

    def add_item(self, cart_id: str, product: Product, quantity: int = 1) -> None:
        self.get(cart_id).add(product, quantity)

    def remove_item(self, cart_id: str, sku: str) -> bool:
        return self.get(cart_id).remove(sku)

    def merge(self, src_id: str, dst_id: str) -> Cart:
        """Fold the source cart into the destination cart and drop the source."""
        src = self.get(src_id)
        dst = self.get(dst_id)
        if src is dst:
            raise CartError("cannot merge a cart with itself")
        for item in src.items:
            dst.add(item.product, item.quantity)
        del self._carts[src_id]
        return dst

    @property
    def active_carts(self) -> int:
        return len(self._carts)

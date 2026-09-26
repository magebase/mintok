"""Append-only purchase history with simple aggregate statistics."""

from __future__ import annotations

from shopcart.models import Order


class PurchaseHistory:
    """Remembers placed orders so a storefront can show spending stats."""

    def __init__(self) -> None:
        self._orders: list[Order] = []

    def record(self, order: Order) -> None:
        self._orders.append(order)

    def __len__(self) -> int:
        return len(self._orders)

    def orders_for(self, sku: str) -> list[Order]:
        """All recorded orders containing the given SKU."""
        return [o for o in self._orders if any(line[0] == sku for line in o.lines)]

    def total_spent(self) -> int:
        return sum(o.total_cents for o in self._orders)

    def units_sold(self, sku: str) -> int:
        total = 0
        for order in self._orders:
            for line_sku, _name, qty, _price in order.lines:
                if line_sku == sku:
                    total += qty
        return total

    def top_product(self) -> str | None:
        """SKU with the most units sold, or None with no history."""
        counts: dict[str, int] = {}
        for order in self._orders:
            for sku, _name, qty, _price in order.lines:
                counts[sku] = counts.get(sku, 0) + qty
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

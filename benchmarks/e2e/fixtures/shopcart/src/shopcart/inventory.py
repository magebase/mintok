"""In-memory stock tracking with reserve/release semantics."""

from __future__ import annotations

DEFAULT_STOCK: dict[str, int] = {
    "BOOK-001": 25,
    "BOOK-002": 8,
    "MUG-100": 60,
    "TEE-L": 14,
}


class OutOfStockError(Exception):
    """Raised when a reservation exceeds available stock."""


class Inventory:
    """Tracks on-hand units per SKU."""

    def __init__(self, initial: dict[str, int] | None = None) -> None:
        self.stock: dict[str, int] = dict(DEFAULT_STOCK if initial is None else initial)

    def add_stock(self, sku: str, units: int) -> None:
        if units <= 0:
            raise ValueError("units must be positive")
        self.stock[sku] = self.stock.get(sku, 0) + units

    def available(self, sku: str) -> int:
        return self.stock.get(sku, 0)

    def reserve(self, sku: str, units: int) -> bool:
        """Reserve units if available; return False when short."""
        if units <= 0:
            raise ValueError("units must be positive")
        current = self.available(sku)
        if current < units:
            return False
        self.stock[sku] = current - units
        return True

    def release(self, sku: str, units: int) -> None:
        """Return reserved units to stock."""
        if units <= 0:
            raise ValueError("units must be positive")
        self.stock[sku] = self.stock.get(sku, 0) + units

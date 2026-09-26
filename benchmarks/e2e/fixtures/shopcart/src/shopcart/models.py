"""Core domain models for carts and orders."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass
class Product:
    """A catalog product. Prices are integer cents to avoid float drift."""

    sku: str
    name: str
    unit_price_cents: int
    weight_grams: int = 0

    def __post_init__(self) -> None:
        if self.unit_price_cents < 0:
            raise ValueError("unit price cannot be negative")


@dataclass
class CartItem:
    product: Product
    quantity: int

    @property
    def line_total_cents(self) -> int:
        return self.product.unit_price_cents * self.quantity


@dataclass
class Cart:
    cart_id: str
    items: list[CartItem] = field(default_factory=list)

    def add(self, product: Product, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        for item in self.items:
            if item.product.sku == product.sku:
                item.quantity += quantity
                return
        self.items.append(CartItem(product=product, quantity=quantity))

    def remove(self, sku: str) -> bool:
        before = len(self.items)
        self.items = [i for i in self.items if i.product.sku != sku]
        return len(self.items) < before

    def item_for(self, sku: str) -> CartItem | None:
        for item in self.items:
            if item.product.sku == sku:
                return item
        return None

    @property
    def total_units(self) -> int:
        return sum(i.quantity for i in self.items)


class OrderStatus(enum.Enum):
    DRAFT = "draft"
    PLACED = "placed"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


@dataclass
class Order:
    order_id: str
    lines: list[tuple[str, str, int, int]]  # (sku, name, qty, unit_price_cents)
    total_cents: int
    status: OrderStatus = OrderStatus.PLACED

    @property
    def line_count(self) -> int:
        return len(self.lines)

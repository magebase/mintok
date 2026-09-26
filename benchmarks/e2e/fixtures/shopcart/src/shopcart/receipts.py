"""Human-readable order receipts."""

from __future__ import annotations

from shopcart.currency import format_money
from shopcart.models import Order


def render_receipt(order: Order) -> str:
    """Render an order as a fixed-width text receipt."""
    lines = [f"Receipt {order.order_id}", "-" * 32]
    for sku, name, qty, unit_price in order.lines:
        line_total = qty * unit_price
        lines.append(f"{name:<18} x{qty:<3} {format_money(line_total):>10}  ({sku})")
    lines.append("-" * 32)
    lines.append(f"{'TOTAL':<18}      {format_money(order.total_cents):>10}")
    return "\n".join(lines)

"""Money formatting helpers. All amounts are integer cents internally."""


def format_money(cents: int) -> str:
    """Render integer cents as a dollar string, e.g. 1234 -> "$12.34"."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"


def parse_money(text: str) -> int:
    """Parse a dollar amount such as "$12.34" or "12.34" into integer cents."""
    cleaned = text.strip().lstrip("$").replace(",", "")
    if not cleaned:
        raise ValueError("empty money string")
    dollars, _, fraction = cleaned.partition(".")
    if not dollars.isdigit() or (fraction and not fraction.isdigit()):
        raise ValueError(f"unparseable money string: {text!r}")
    if len(fraction) > 2:
        raise ValueError("more than two decimal places")
    fraction = (fraction or "0").ljust(2, "0")
    return int(dollars) * 100 + int(fraction)


def round_half_up(amount: float) -> int:
    """Round a fractional cent amount half away from zero (positive values)."""
    import math

    return math.floor(amount + 0.5)

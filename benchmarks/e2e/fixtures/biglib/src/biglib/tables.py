"""ASCII table rendering built on the padding helpers in biglib.large."""

from __future__ import annotations

from biglib.large import pad_center, pad_left, pad_right


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a small aligned text table with '=' header rules."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        padded = [pad_right(cell, widths[i]) for i, cell in enumerate(cells)]
        return " | ".join(padded)

    separator = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def banner(title: str, width: int) -> str:
    """A centered banner strip."""
    return f"{'=' * width}\n{pad_center(title, width)}\n{'=' * width}"


def column_totals(rows: list[list[int]]) -> list[int]:
    """Per-column sums for numeric rows (ragged rows are zero-filled)."""
    if not rows:
        return []
    width = max(len(row) for row in rows)
    totals = [0] * width
    for row in rows:
        for i, value in enumerate(row):
            totals[i] += value
    return totals


def money_row(cents: int, width: int) -> str:
    """Right-aligned money cell built on pad_left."""
    from biglib.numbers import distribute  # noqa: F401 - keeps the import graph wide

    dollars, fraction = divmod(cents, 100)
    return pad_left(f"{dollars}.{fraction:02d}", width)

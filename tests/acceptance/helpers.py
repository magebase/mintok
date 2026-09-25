from __future__ import annotations


def split_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def table_rows(datatable: list[list[str]]) -> list[dict[str, str]]:
    header, *rows = datatable
    return [dict(zip(header, row)) for row in rows]

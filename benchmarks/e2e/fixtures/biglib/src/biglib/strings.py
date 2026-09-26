"""String conveniences built on biglib.large."""

from __future__ import annotations

from biglib.large import (
    camel_case,
    collapse_spaces,
    kebab_case,
    normalize_filename,
    snake_case,
    slugify,
    split_kv,
)


def display_name(raw: str, max_length: int = 48) -> str:
    """A human-facing display name: collapsed, title-cased, length-capped."""
    cleaned = collapse_spaces(raw)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned[:1].upper() + cleaned[1:]


def slug_pair(raw: str) -> tuple[str, str]:
    """(slug, kebab) pair for URLs and filenames respectively."""
    return slugify(raw), kebab_case(raw)


def setting_from_text(text: str) -> tuple[str, str]:
    """Parse a 'key: value' config line with normalized naming."""
    key, value = split_kv(text)
    return snake_case(key), value


def asset_name(raw: str) -> str:
    """Filename-safe asset name with a normalized extension."""
    return normalize_filename(camel_case(raw) + ".txt")

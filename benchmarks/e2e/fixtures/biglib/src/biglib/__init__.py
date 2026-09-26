"""biglib: a synthetic utility library whose text tools live in one oversized module."""

from biglib.large import (
    camel_case,
    collapse_spaces,
    kebab_case,
    pad_center,
    snake_case,
    slugify,
    truncate,
    truncate_with_ellipsis,
    wrap_text,
)

__all__ = [
    "camel_case",
    "collapse_spaces",
    "kebab_case",
    "pad_center",
    "snake_case",
    "slugify",
    "truncate",
    "truncate_with_ellipsis",
    "wrap_text",
]

"""Token estimation. Swap in a model-specific tokenizer per backend when measured."""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0

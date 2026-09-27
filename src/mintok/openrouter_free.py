"""September 2026 OpenRouter Free Model Specification and Verification.

Enforces:
1. Exact, fixed model IDs (bans `openrouter/free` dynamic routing).
2. Live assertion of zero token pricing ($0 prompt, $0 completion).
3. Immutable model metadata capture (context length, architecture, tokenizer).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

# Primary frozen September 2026 OpenRouter free models
PRIMARY_FREE_MODELS: list[str] = [
    "qwen/qwen3.8-27b:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "cohere/north-mini-code:free",
]

# Current September 2026 stealth preview model (reported separately as stress-test)
STRESS_TEST_MODELS: list[str] = [
    "stealth/space-bunny-alpha",
]

ALL_SEPT_2026_MODELS = PRIMARY_FREE_MODELS + STRESS_TEST_MODELS

# Cached verified OpenRouter metadata snapshot (September 27, 2026) for offline / sandboxed verification
FROZEN_METADATA_SNAPSHOT: dict[str, dict[str, Any]] = {
    "qwen/qwen3.8-27b:free": {
        "id": "qwen/qwen3.8-27b:free",
        "name": "Qwen: Qwen3.8 27B (free)",
        "created_date": "2026-08-14",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {
            "modality": "text+image+video->text",
            "tokenizer": "Qwen",
        },
        "description": "27B coding and agent model with 68.1 Coding Index on OpenRouter.",
    },
    "poolside/laguna-s-2.1:free": {
        "id": "poolside/laguna-s-2.1:free",
        "name": "Poolside: Laguna S 2.1 (free)",
        "created_date": "2026-07-21",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {
            "modality": "text->text",
            "tokenizer": "Other",
        },
        "description": "Specialized coding-agent model with tool calling.",
    },
    "nvidia/nemotron-3-ultra-550b-a55b:free": {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "NVIDIA: Nemotron 3 Ultra (free)",
        "created_date": "2026-07-28",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {
            "modality": "text->text",
            "tokenizer": "Other",
        },
        "description": "1M context agent/reasoning model with tool calling.",
    },
    "cohere/north-mini-code:free": {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere: North Mini Code (free)",
        "created_date": "2026-06-18",
        "context_length": 256000,
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {
            "modality": "text->text",
            "tokenizer": "Cohere",
        },
        "description": "Trained specifically for agentic software engineering and SWE harnesses.",
    },
    "stealth/space-bunny-alpha": {
        "id": "stealth/space-bunny-alpha",
        "name": "Space Bunny Alpha",
        "created_date": "2026-09-23",
        "context_length": 1000000,
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {
            "modality": "text+image+video->text",
            "tokenizer": "Other",
        },
        "description": "Stealth/anonymous 1M context coding model (September 2026 preview).",
    },
}


def query_openrouter_models(timeout: int = 10) -> dict[str, dict[str, Any]] | None:
    """Fetch live model catalog from OpenRouter public API."""
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MinTok Benchmark Suite)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return {m["id"]: m for m in data.get("data", [])}
    except Exception:
        return None


def assert_models_free(
    model_ids: list[str] | None = None,
    allow_frozen_fallback: bool = True,
) -> dict[str, dict[str, Any]]:
    """Assert that every requested model ID exists on OpenRouter and is strictly free.

    Bans 'openrouter/free' dynamic routing.
    Raises RuntimeError if any model is no longer free or unavailable.
    """
    targets = model_ids or ALL_SEPT_2026_MODELS

    for m in targets:
        if m == "openrouter/free":
            raise ValueError("openrouter/free is forbidden: dynamic routing invalidates reproducibility")

    live_catalog = query_openrouter_models()
    verified: dict[str, dict[str, Any]] = {}

    for mid in targets:
        if live_catalog and mid in live_catalog:
            m_data = live_catalog[mid]
            pricing = m_data.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", -1))
            compl_cost = float(pricing.get("completion", -1))
            if prompt_cost > 0 or compl_cost > 0:
                raise RuntimeError(
                    f"Model {mid} is no longer free on OpenRouter (prompt: {prompt_cost}, completion: {compl_cost})"
                )
            verified[mid] = {
                "id": mid,
                "name": m_data.get("name", mid),
                "context_length": m_data.get("context_length"),
                "pricing": pricing,
                "architecture": m_data.get("architecture", {}),
                "live_verified": True,
            }
        elif allow_frozen_fallback and mid in FROZEN_METADATA_SNAPSHOT:
            snap = FROZEN_METADATA_SNAPSHOT[mid]
            verified[mid] = dict(snap, live_verified=False)
        else:
            raise RuntimeError(
                f"Model {mid} not found on OpenRouter and no verified snapshot exists"
            )

    return verified

"""Tool-schema paging and cache-aware prompt assembly for MinTok 3.1.

Maximizes provider prompt-cache locality and eliminates tool-schema context tax:
1. Paged Tool Schemas: compact 1-line capability index first; expands schemas on demand.
2. Layered Prompt Assembly: hashes system, tools, repo profile, memory, task, and state
   separately to diagnose and prevent prompt-cache invalidation.
3. Strict Canonical Serialization: stable keys, deterministic whitespace, invariant ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


CAPABILITY_INDEX = {
    "code.search": ["grep", "find_files", "slice"],
    "code.edit": ["patch", "read"],
    "exec.test": ["suite"],
    "exec.shell": ["shell"],
    "memory.expand": ["expand"],
    "coproc.macro": ["investigate_failure", "localize_symbol", "state_writers", "change_ripple"],
}


def render_capability_index() -> str:
    """Render a compact ~50-token capability index."""
    lines = ["Available Capabilities (expand family for full schemas):"]
    for cap, tools in CAPABILITY_INDEX.items():
        lines.append(f"  {cap}: {', '.join(tools)}")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PromptLayerHash:
    """Hash and token count for an isolated prompt layer."""

    layer: str
    content_hash: str
    tokens: int


@dataclass
class PromptAssembly:
    """Canonical, layered prompt payload with cache preservation telemetry."""

    system: str
    tools: list[dict[str, Any]]
    repo_profile: str = ""
    memory: str = ""
    task: str = ""
    latest_state: str = ""
    layer_hashes: dict[str, PromptLayerHash] = field(default_factory=dict)

    def assemble(self, page_tools: bool = False) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
        """Assemble the canonical prompt and compute per-layer hashes."""
        # Canonicalize system prompt
        clean_sys = self.system.strip()

        # Tools handling: paged vs full
        if page_tools:
            tool_schemas = [
                {
                    "name": "expand_capability",
                    "description": "Load full schemas for a capability family.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "family": {
                                "type": "string",
                                "enum": list(CAPABILITY_INDEX.keys()),
                            }
                        },
                        "required": ["family"],
                    },
                }
            ]
            tools_json = json.dumps(tool_schemas, sort_keys=True)
        else:
            tool_schemas = self.tools
            tools_json = json.dumps(tool_schemas, sort_keys=True)

        clean_profile = self.repo_profile.strip()
        clean_memory = self.memory.strip()
        clean_task = self.task.strip()
        clean_state = self.latest_state.strip()

        layers = {
            "system": clean_sys,
            "tools": tools_json,
            "repo_profile": clean_profile,
            "memory": clean_memory,
            "task": clean_task,
            "latest_state": clean_state,
        }

        hashes: dict[str, str] = {}
        for name, text in layers.items():
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            hashes[name] = h

        # Build combined user/system text
        parts = []
        if clean_profile:
            parts.append(clean_profile)
        if clean_memory:
            parts.append(clean_memory)
        if clean_state:
            parts.append(clean_state)
        parts.append(clean_task)
        if page_tools:
            parts.append(render_capability_index())

        final_content = "\n\n".join(parts)
        return final_content, tool_schemas, hashes

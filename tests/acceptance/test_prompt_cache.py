"""Unit tests for tool-schema paging and prompt layer cache assembly."""

from __future__ import annotations

import json
from mintok.prompt_cache import (
    CAPABILITY_INDEX,
    PromptAssembly,
    render_capability_index,
)


def test_capability_index_rendering():
    index_text = render_capability_index()
    assert "Available Capabilities" in index_text
    assert "code.search" in index_text
    assert "exec.shell" in index_text
    assert "memory.expand" in index_text


def test_prompt_assembly_full_tools():
    tools = [
        {"name": "shell", "description": "run shell", "input_schema": {"type": "object"}},
        {"name": "read", "description": "read file", "input_schema": {"type": "object"}},
    ]
    assembly = PromptAssembly(
        system="You are an expert engineer.",
        tools=tools,
        repo_profile="[repo profile: pytest]",
        memory="known: foo()",
        task="Fix the bug",
        latest_state="State: 1 failure",
    )
    content, out_tools, hashes = assembly.assemble(page_tools=False)
    assert len(out_tools) == 2
    assert "system" in hashes
    assert "tools" in hashes
    assert "repo_profile" in hashes
    assert "memory" in hashes
    assert "latest_state" in hashes
    assert "task" in hashes
    assert "[repo profile: pytest]" in content
    assert "Fix the bug" in content


def test_prompt_assembly_paged_tools():
    tools = [
        {"name": "shell", "description": "run shell", "input_schema": {"type": "object"}},
        {"name": "read", "description": "read file", "input_schema": {"type": "object"}},
    ]
    assembly = PromptAssembly(
        system="You are an expert engineer.",
        tools=tools,
        task="Fix the bug",
    )
    content, out_tools, hashes = assembly.assemble(page_tools=True)
    # When paged, only 1 meta-tool (expand_capability) is exposed initially
    assert len(out_tools) == 1
    assert out_tools[0]["name"] == "expand_capability"
    assert "Available Capabilities" in content


def test_prompt_assembly_hash_stability():
    tools = [{"name": "shell", "input_schema": {}}]
    assembly1 = PromptAssembly(system="sys", tools=tools, task="do work")
    assembly2 = PromptAssembly(system="sys", tools=tools, task="do work")

    _, _, hashes1 = assembly1.assemble()
    _, _, hashes2 = assembly2.assemble()

    assert hashes1 == hashes2

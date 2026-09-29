"""Acceptance test steps for repository execution profiles."""

from __future__ import annotations

import json
from pathlib import Path
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.repo_profile import RepoProfile, scan_repo_profile
from mintok.tokens import estimate_tokens

scenarios("repo_profile.feature")


@given("a repository with structure:", target_fixture="ctx")
def given_repo_structure(tmp_path: Path, docstring: str):
    root = tmp_path / "repo1"
    root.mkdir()
    for line in docstring.splitlines():
        line = line.strip()
        if not line:
            continue
        p = root / line
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return {"root": root}


@given("a monorepo with 5 packages:", target_fixture="ctx")
def given_monorepo_structure(tmp_path: Path, docstring: str):
    root = tmp_path / "monorepo"
    root.mkdir()
    for line in docstring.splitlines():
        line = line.strip()
        if not line:
            continue
        p = root / line
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return {"root": root}


@given(parsers.parse('a cached repository profile for "{name}":'), target_fixture="ctx")
def given_cached_profile(name: str, docstring: str):
    profile = RepoProfile(
        repo_name=name,
        package_manager="pip",
        test_runner="pytest",
        test_command="pytest -q",
        packages=["hyp3_sdk"],
        key_directories=["src/hyp3_sdk", "tests"],
        complexity_score=0.25,
        recommended_strategy="compressed-first",
    )
    return {"profile": profile}


@given(parsers.parse('a repository profile for "{name}"'), target_fixture="ctx")
def given_simple_profile(name: str):
    profile = RepoProfile(
        repo_name=name,
        package_manager="setuptools",
        test_runner="pytest",
        test_command="pytest -q",
        packages=["msrest", "msrest.serialization"],
        key_directories=["msrest", "tests"],
        complexity_score=0.45,
        recommended_strategy="compressed-first",
    )
    return {"profile": profile}


@when("the repository execution profile is scanned")
def when_profile_scanned(ctx):
    profile = scan_repo_profile(ctx["root"])
    ctx["profile"] = profile


@when("the profile context is formatted for the agent")
def when_context_formatted(ctx):
    formatted = ctx["profile"].render_context()
    ctx["formatted"] = formatted


@when("the profile is serialized to JSON and reloaded")
def when_serialized_and_reloaded(ctx, tmp_path: Path):
    orig = ctx["profile"]
    p_path = tmp_path / "profile.json"
    orig.save(p_path)
    reloaded = RepoProfile.load(p_path)
    ctx["reloaded"] = reloaded


@then(parsers.parse('the detected package manager is "{pkg_mgr}"'))
def then_detected_pkg_mgr(ctx, pkg_mgr: str):
    assert ctx["profile"].package_manager == pkg_mgr


@then(parsers.parse('the detected test runner is "{runner}"'))
def then_detected_runner(ctx, runner: str):
    assert ctx["profile"].test_runner == runner


@then(parsers.parse('the key directories include "{d1}" and "{d2}"'))
def then_key_dirs_include(ctx, d1: str, d2: str):
    dirs = ctx["profile"].key_directories
    assert d1 in dirs
    assert d2 in dirs


@then(parsers.parse('the recommended strategy is "{strategy}"'))
def then_recommended_strategy(ctx, strategy: str):
    assert ctx["profile"].recommended_strategy == strategy


@then(parsers.parse("the monorepo package count is at least {count:d}"))
def then_package_count_at_least(ctx, count: int):
    assert len(ctx["profile"].packages) >= count


@then(parsers.parse("the complexity score exceeds {threshold:f}"))
def then_complexity_exceeds(ctx, threshold: float):
    assert ctx["profile"].complexity_score > threshold


@then(parsers.parse("the profile token estimate is under {max_tokens:d} tokens"))
def then_token_estimate_under(ctx, max_tokens: int):
    toks = estimate_tokens(ctx["formatted"])
    assert toks < max_tokens


@then(parsers.parse('the profile context contains "{substr}"'))
def then_context_contains(ctx, substr: str):
    assert substr in ctx["formatted"]


@then("the reloaded profile matches the original profile")
def then_reloaded_matches(ctx):
    assert ctx["reloaded"].to_dict() == ctx["profile"].to_dict()

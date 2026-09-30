"""Acceptance test steps for runtime diagnostics: doctor and inspect-run."""

from __future__ import annotations

import json
from pathlib import Path
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.cli import main as mintok_main
from mintok.doctor import run_doctor_checks

scenarios("runtime_diagnostics.feature")


@given("a clean repository with structure:", target_fixture="ctx")
def given_clean_repo(tmp_path: Path, docstring: str):
    root = tmp_path / "diag_repo"
    root.mkdir()
    for line in docstring.splitlines():
        line = line.strip()
        if not line:
            continue
        p = root / line
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return {"root": root}


@when('running "mintok doctor" on the repository')
def when_run_doctor(ctx, capsys):
    ret = mintok_main(["doctor", str(ctx["root"])])
    out = capsys.readouterr().out
    ctx["doctor_ret"] = ret
    ctx["doctor_out"] = out


@then("the doctor report passes")
def then_doctor_passes(ctx):
    assert ctx["doctor_ret"] == 0
    assert "Status:     PASSED" in ctx["doctor_out"]


@then("the report includes tool ABI budget verification under 300 tokens")
def then_doctor_abi_budget(ctx):
    assert "Tool ABI Budget" in ctx["doctor_out"]
    assert "limit <= 300 tokens" in ctx["doctor_out"]


@then(parsers.parse('the report confirms policy compatibility for "{policy}"'))
def then_doctor_confirms_policy(ctx, policy: str):
    assert f"Policy '{policy}' is CURRENT_EXPERIMENTAL" in ctx["doctor_out"]


@when(parsers.parse('running "mintok doctor" with policy "{policy}"'))
def when_run_doctor_with_policy(ctx, policy: str, capsys):
    ret = mintok_main(["doctor", str(ctx["root"]), "--policy", policy])
    out = capsys.readouterr().out
    ctx["doctor_ret"] = ret
    ctx["doctor_out"] = out


@then("the doctor report fails")
def then_doctor_fails(ctx):
    assert ctx["doctor_ret"] == 1
    assert "Status:     FAILED" in ctx["doctor_out"]


@then(parsers.parse('the report flags policy "{policy}" as obsolete'))
def then_flags_obsolete(ctx, policy: str):
    assert f"Policy '{policy}' is LEGACY_DEVELOPMENT_ONLY (obsolete)" in ctx["doctor_out"]


@given("a trajectory JSON file with 3 turns", target_fixture="ctx")
def given_trajectory_file(tmp_path: Path):
    traj_path = tmp_path / "sample_trajectory.json"
    data = {
        "task_id": "test_task_1",
        "turns": [
            {
                "turn": 1,
                "action": "virtualize",
                "t_fresh": 500,
                "t_cached": 1000,
                "t_out": 100,
                "t_reasoning": 50,
                "cost_usd": 0.005,
                "progress": "indexed",
            },
            {
                "turn": 2,
                "action": "grep",
                "t_fresh": 4000,
                "t_cached": 2000,
                "t_out": 200,
                "t_reasoning": 0,
                "cost_usd": 0.02,
                "progress": "found symbol",
            },
            {
                "turn": 3,
                "action": "verify",
                "t_fresh": 1200,
                "t_cached": 2000,
                "t_out": 50,
                "t_reasoning": 0,
                "cost_usd": 0.008,
                "progress": "failed: assertion error",
            },
        ],
    }
    traj_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"traj_path": traj_path}


@when('running "mintok inspect-run" on the trajectory')
def when_run_inspect(ctx, capsys):
    ret = mintok_main(["inspect-run", str(ctx["traj_path"])])
    out = capsys.readouterr().out
    ctx["cli_ret"] = ret
    ctx["cli_out"] = out


@then(parsers.parse('the CLI stdout contains "{text}"'))
def then_cli_contains(ctx, text: str):
    assert text in ctx["cli_out"]

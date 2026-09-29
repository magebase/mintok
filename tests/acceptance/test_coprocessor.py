"""Acceptance test steps for semantic coprocessor and macro-actions."""

from __future__ import annotations

from pathlib import Path
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.coprocessor import SemanticCoprocessor

scenarios("coprocessor.feature")


@given(parsers.parse('a Python repository with a failing function in "{fpath}":'), target_fixture="ctx")
def given_repo_with_failing_function(tmp_path: Path, fpath: str, docstring: str):
    root = tmp_path / "repo_coproc"
    root.mkdir()
    target = root / fpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(docstring, encoding="utf-8")
    coproc = SemanticCoprocessor(repo_root=root)
    return {"root": root, "coproc": coproc, "target_fpath": fpath}


@given(parsers.parse('a test in "{fpath}":'))
def given_test_file(ctx, fpath: str, docstring: str):
    t_file = ctx["root"] / fpath
    t_file.parent.mkdir(parents=True, exist_ok=True)
    t_file.write_text(docstring, encoding="utf-8")


@given(parsers.parse('a repository with symbol "{symbol}" defined in "{fpath}"'), target_fixture="ctx")
def given_symbol_defined(tmp_path: Path, symbol: str, fpath: str):
    root = tmp_path / "repo_loc"
    root.mkdir()
    p = root / fpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"def {symbol}(username, password):\n    return True\n", encoding="utf-8")
    coproc = SemanticCoprocessor(repo_root=root)
    return {"root": root, "coproc": coproc, "symbol": symbol, "def_path": fpath}


@given(parsers.parse('"{caller_path}" calls "{symbol}"'))
def given_caller_calls_symbol(ctx, caller_path: str, symbol: str):
    p = ctx["root"] / caller_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"from auth import {symbol}\ndef handle_req():\n    return {symbol}('admin', '123')\n", encoding="utf-8")


@given(parsers.parse('an existing symbol "{sig}" in "{fpath}"'), target_fixture="ctx")
def given_existing_symbol(tmp_path: Path, sig: str, fpath: str):
    root = tmp_path / "repo_patch"
    root.mkdir()
    sym_name = sig.split("(")[0]
    old_code = f"def {sym_name}(record):\n    return str(record)\n"
    p = root / fpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(old_code, encoding="utf-8")
    coproc = SemanticCoprocessor(repo_root=root)
    return {"root": root, "coproc": coproc, "old_code": old_code, "sym_name": sym_name}


@when(parsers.parse('a test failure occurs with traceback pointing to "{target_frame}"'))
def when_failure_occurs(ctx, target_frame: str):
    fpath, lno = target_frame.split(":")
    tb = (
        f'Traceback (most recent call last):\n'
        f'  File "tests/test_service.py", line 3, in test_empty\n'
        f'    process_data("")\n'
        f'  File "{fpath}", line {lno}, in process_data\n'
        f'    raise ValueError("empty data")\n'
        f'ValueError: empty data\n'
    )
    ctx["traceback"] = tb


@when(parsers.parse('macro-action "{action}" is executed'))
def when_macro_executed(ctx, action: str):
    if action == "investigate_failure":
        packet = ctx["coproc"].investigate_failure(ctx["root"], ctx["traceback"])
        ctx["packet"] = packet


@when(parsers.parse('macro-action "{action}" is invoked for "{symbol}"'))
def when_macro_invoked_for_symbol(ctx, action: str, symbol: str):
    if action == "localize_symbol":
        packet = ctx["coproc"].localize_symbol(ctx["root"], symbol)
        ctx["packet"] = packet


@when(parsers.parse('a patch modifies internal logic of "{sym_name}" without changing signature'))
def when_patch_modifies_body(ctx, sym_name: str):
    new_code = f"def {sym_name}(record):\n    if record is None:\n        return ''\n    return str(record)\n"
    assessment = ctx["coproc"].assess_patch(ctx["old_code"], new_code, sym_name)
    ctx["assessment"] = assessment


@then(parsers.parse('the resulting evidence packet targets symbol "{symbol}"'))
def then_targets_symbol(ctx, symbol: str):
    assert ctx["packet"].target_symbol == symbol


@then(parsers.parse('the evidence packet target file is "{fpath}"'))
def then_target_file(ctx, fpath: str):
    assert ctx["packet"].target_file == fpath


@then(parsers.parse("the evidence packet token count is under {limit:d} tokens"))
def then_token_count_under(ctx, limit: int):
    assert ctx["packet"].tokens < limit


@then(parsers.parse('the evidence packet includes the slice excerpt of "{symbol}"'))
def then_includes_slice(ctx, symbol: str):
    assert symbol in ctx["packet"].slice_excerpt


@then(parsers.parse('the symbol packet identifies definition in "{fpath}"'))
def then_identifies_def(ctx, fpath: str):
    assert ctx["packet"].file_path == fpath


@then(parsers.parse('the symbol packet identifies caller "{caller}"'))
def then_identifies_caller(ctx, caller: str):
    assert caller in ctx["packet"].callers


@then(parsers.parse("the packet size is under {limit:d} tokens"))
def then_packet_size_under(ctx, limit: int):
    assert ctx["packet"].tokens < limit


@then(parsers.parse('the patch assessment classifies the change as "{change_type}"'))
def then_change_type(ctx, change_type: str):
    assert ctx["assessment"].change_type == change_type


@then("interface compatibility is preserved")
def then_interface_preserved(ctx):
    assert ctx["assessment"].interface_compatible is True

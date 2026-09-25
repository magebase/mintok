from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import parsers, scenarios, then, when

from mintok.compiler import compile_repository
from tests.acceptance.helpers import table_rows

scenarios("hash_invalidation.feature")

# Synthetic billing fixture; each mutation is one class of source edit.
BASE = '''class Gateway:
    def refund(self, payment_id, amount):
        return True


class Payment:
    def audit(self):
        return True

    def refund(self, amount: int) -> bool:
        """Refund part or all of a captured payment."""
        if amount <= 0:
            raise ValueError()
        Gateway().refund(self.id, amount)
        self.refunded_amount = amount
        return True
'''

MUTATIONS: dict[str, str] = {
    "comment": BASE.replace(
        "class Gateway:", "# refund policy: see the payments handbook\nclass Gateway:"
    ),
    "docstring": BASE.replace(
        '"""Refund part or all of a captured payment."""',
        '"""Refund captured payments."""',
    ),
    "formatting": BASE.replace("if amount <= 0:", "if amount<=0:"),
    "rename_local": BASE.replace(
        "Gateway().refund(self.id, amount)",
        "payment_total = amount\n        Gateway().refund(self.id, payment_total)",
    ),
    "reorder_condition": BASE.replace("if amount <= 0:", "if 0 >= amount:"),
    "add_print": BASE.replace(
        "Gateway().refund(self.id, amount)",
        'print("refunding payment")\n        Gateway().refund(self.id, amount)',
    ),
    "new_exception": BASE.replace(
        "raise ValueError()", "raise ValueError()\n            raise RuntimeError()"
    ),
    "change_return_type": BASE.replace("-> bool", "-> int"),
    "new_call": BASE.replace(
        "Gateway().refund(self.id, amount)",
        "self.audit()\n        Gateway().refund(self.id, amount)",
    ),
    "new_write": BASE.replace(
        "self.refunded_amount = amount",
        "self.refunded_amount = amount\n        self.refund_count = amount",
    ),
    "retarget_write": BASE.replace("self.refunded_amount = amount", "self.refund_total = amount"),
}


@when("every named mutation is applied and recompiled")
def apply_mutations(ctx: SimpleNamespace, repo: Path) -> None:
    assert ctx.ir is not None, "background must compile the baseline first"
    baseline_text = (repo / "billing.py").read_text()
    ctx.mutations = {}
    try:
        for name, mutated in MUTATIONS.items():
            assert mutated != baseline_text, f"mutation {name} is a no-op"
            (repo / "billing.py").write_text(mutated)
            ir = compile_repository(repo)
            assert not ir.diagnostics, (name, ir.diagnostics)
            old = ctx.ir.symbols["billing:Payment.refund"]
            new = ir.symbols["billing:Payment.refund"]
            ctx.mutations[name] = {
                "body": old.body_hash != new.body_hash,
                "interface": old.interface_hash != new.interface_hash,
            }
    finally:
        (repo / "billing.py").write_text(baseline_text)


@then("invalidation matches expectation:")
def invalidation_matches(ctx: SimpleNamespace, datatable: list[list[str]]) -> None:
    rows = table_rows(datatable)
    assert set(ctx.mutations) == {row["mutation"] for row in rows}, ctx.mutations
    for row in rows:
        observed = ctx.mutations[row["mutation"]]
        assert observed["body"] == (row["body_changes"] == "yes"), (row["mutation"], observed)
        assert observed["interface"] == (row["interface_changes"] == "yes"), (
            row["mutation"],
            observed,
        )

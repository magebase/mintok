from __future__ import annotations

from types import SimpleNamespace

from pytest_bdd import given, parsers, scenarios, then, when

from mintok.metrics import OCU_USD, RunRecord, optimization_compute_units

scenarios("ocu.feature")


@given(
    parsers.parse(
        "a meter over {n:d} run(s) at ${f_usd:g} frontier, ${l_usd:g} local, ${i_usd:g} indexing,"
        " ${s_usd:g} storage, and ${c_usd:g} cpu per run"
    )
)
def meter(
    ctx: SimpleNamespace,
    n: int,
    f_usd: float,
    l_usd: float,
    i_usd: float,
    s_usd: float,
    c_usd: float,
) -> None:
    ctx.meter = [
        RunRecord(
            task_id=f"t{k}",
            arm="arm",
            solved=True,
            frontier_usd=f_usd,
            local_usd=l_usd,
            indexing_usd=i_usd,
            storage_usd=s_usd,
            cpu_usd=c_usd,
        )
        for k in range(1, n + 1)
    ]


@when("the metering is computed")
def compute_metering(ctx: SimpleNamespace) -> None:
    ctx.total_usd = sum(r.total_usd for r in ctx.meter)
    ctx.ocu = optimization_compute_units(ctx.total_usd)


@then(parsers.parse('the total cost per run is "${cost:g}"'))
def cost_per_run(ctx: SimpleNamespace, cost: float) -> None:
    assert round(ctx.total_usd / len(ctx.meter), 2) == cost


@then(parsers.parse('the metered total is "{ocus:g}" OCUs'))
def metered_total(ctx: SimpleNamespace, ocus: float) -> None:
    assert round(ctx.ocu, 2) == ocus


@then(parsers.parse('frontier spend of "${usd:g}" meters to exactly "{ocus:g}" OCUs'))
def no_markup(usd: float, ocus: float) -> None:
    assert OCU_USD == 0.01
    assert optimization_compute_units(usd) == ocus

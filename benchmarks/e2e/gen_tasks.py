"""Stratified 120-task benchmark generator over the synthetic fixture repos.

Task schema mirrors benchmarks/e2e/tasks.py exactly: ``id``, ``klass``,
``instruction``, ``check``. Checks are plain Python strings evaluated with
``root`` bound to the task copy; each inserts the fixture's ``src`` on
sys.path, runs deterministic asserts, and sets ``ok``. The fixture's own
pytest suite is scored separately by the harness (suite_ok).

Strata (klass):
  simple_lookup             15   single-function fixes, docstrings, tweaks
  cross_file_bug            20   fix in module A, asserted via module B
  feature_addition          20   new function/method/flag with clear spec
  refactor                  15   extract helper / rename / delete dead code
  api_signature_propagation 20   new parameter or signature change + callers
  schema_or_framework_change 15  dataclass field/type + serialization users
  large_file_navigation     15   locate + edit inside biglib/src/biglib/large.py

Subcommands:
  --validate   run every check against a pristine temp copy; all must FAIL
  --solve      apply the bundled reference diffs (fixtures/solutions/) for the
               12 sampled tasks and assert the checks now PASS and the fixture
               suite still passes on the solved copy
  --emit       write benchmarks/e2e/tasks_generated.json
  --run-check TASK_ID ROOT   internal: exec one check, exit 0 iff it passes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
SOLUTIONS = FIXTURES / "solutions"
EMIT_PATH = HERE / "tasks_generated.json"

# Every check gets the fixture's src on sys.path so it never depends on MinTok
# or on harness cwd/PYTHONPATH conventions.
PREAMBLE = (
    "import os, sys\n"
    "sys.path.insert(0, os.path.join(root, 'src'))\n"
)


def _c(body: str) -> str:
    """Assemble a full check snippet: preamble + dedented body + ok flag."""
    return PREAMBLE + textwrap.dedent(body).strip("\n") + "\nok = True\n"


TASKS: list[dict] = [
    # =====================================================================
    # shopcart (23)
    # =====================================================================
    {
        "id": "shopcart-lookup-01",
        "klass": "simple_lookup",
        "repo": "shopcart",
        "instruction": (
            "Make parse_money in src/shopcart/currency.py accept negative amounts: "
            '"-$2.50" and "-2.50" must parse to -250. Positive parsing and error '
            "cases are unchanged."
        ),
        "check": _c(
            """
            from shopcart.currency import parse_money
            assert parse_money("-$2.50") == -250
            assert parse_money("-2.50") == -250
            assert parse_money("$12.34") == 1234
            try:
                parse_money("abc")
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """
        ),
    },
    {
        "id": "shopcart-lookup-02",
        "klass": "simple_lookup",
        "repo": "shopcart",
        "instruction": (
            "Fix round_half_up in src/shopcart/currency.py so negative halves round "
            "away from zero: round_half_up(-2.5) == -3 and round_half_up(-2.4) == -2. "
            "Positive behavior must stay identical."
        ),
        "check": _c(
            """
            from shopcart.currency import round_half_up
            assert round_half_up(-2.5) == -3
            assert round_half_up(-2.4) == -2
            assert round_half_up(2.5) == 3
            assert round_half_up(2.4) == 2
            """
        ),
    },
    {
        "id": "shopcart-lookup-03",
        "klass": "simple_lookup",
        "repo": "shopcart",
        "instruction": (
            "Strengthen Product.__post_init__ in src/shopcart/models.py: raise "
            "ValueError when sku or name is blank. Existing valid products are "
            "unaffected."
        ),
        "check": _c(
            """
            from shopcart.models import Product
            for bad in (("", "Name", 100), ("SKU-1", "   ", 100)):
                try:
                    Product(bad[0], bad[1], bad[2])
                    raise AssertionError(f"expected ValueError for {bad}")
                except ValueError:
                    pass
            p = Product("SKU-1", "Name", 100)
            assert p.unit_price_cents == 100
            """
        ),
    },
    {
        "id": "shopcart-cross-01",
        "klass": "cross_file_bug",
        "repo": "shopcart",
        "instruction": (
            "OrderService.order_summary in src/shopcart/orders.py misuses "
            "pricing.compute_total by passing order.total_cents as the subtotal. Fix "
            "it so 'subtotal' is the sum of the order line totals, 'discount' is the "
            "code discount that was applied, and 'total' equals order.total_cents."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            service = OrderService(Inventory())
            cart = Cart(cart_id="c")
            cart.add(Product("BOOK-001", "Book", 1995), 2)
            order = service.place_order(cart, "US", code="SAVE10")
            summary = service.order_summary(order)
            assert summary["subtotal"] == 3990, summary
            assert summary["discount"] == 399, summary
            assert summary["total"] == order.total_cents, summary
            assert summary["lines"] == 1, summary
            """
        ),
    },
    {
        "id": "shopcart-cross-02",
        "klass": "cross_file_bug",
        "repo": "shopcart",
        "instruction": (
            "Free shipping must be decided on the discounted subtotal, not the raw "
            "cart subtotal. Add an optional discount_cents parameter (default 0) to "
            "shipping_for_cart in src/shopcart/shipping.py, and update "
            "OrderService.place_order in src/shopcart/orders.py to pass the applied "
            "discount when it calls shipping_for_cart."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            from shopcart.shipping import shipping_for_cart
            cart = Cart(cart_id="c")
            cart.add(Product("TEE-L", "Thing", 4000), 2)  # subtotal 8000
            assert shipping_for_cart(cart, "US", discount_cents=800) == 500
            service = OrderService(Inventory())
            order = service.place_order(cart, "US", code="SAVE10")
            # discounted 7200 < threshold, so shipping 500 applies
            assert order.total_cents == 7200 + 594 + 500, order.total_cents
            """
        ),
    },
    {
        "id": "shopcart-cross-03",
        "klass": "cross_file_bug",
        "repo": "shopcart",
        "instruction": (
            "OrderService.place_order in src/shopcart/orders.py reserves inventory "
            "item by item, so a shortage on a later item leaks the earlier "
            "reservations. Roll back every reservation made before raising "
            "OutOfStockError."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory, OutOfStockError
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            inventory = Inventory({"A1": 5, "B2": 1})
            service = OrderService(inventory)
            cart = Cart(cart_id="c")
            cart.add(Product("A1", "Alpha", 100), 5)
            cart.add(Product("B2", "Beta", 200), 2)
            try:
                service.place_order(cart, "US")
                raise AssertionError("expected OutOfStockError")
            except OutOfStockError:
                pass
            assert inventory.available("A1") == 5, inventory.available("A1")
            assert inventory.available("B2") == 1
            """
        ),
    },
    {
        "id": "shopcart-cross-04",
        "klass": "cross_file_bug",
        "repo": "shopcart",
        "instruction": (
            "Placing an order to an unknown region currently raises ValueError from "
            "the shipping module. Make OrderService.place_order in "
            "src/shopcart/orders.py fall back to the 'ROW' regional rate for "
            "unrecognized regions so placement succeeds."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            service = OrderService(Inventory())
            cart = Cart(cart_id="c")
            cart.add(Product("MUG-100", "Mug", 825), 1)
            try:
                order = service.place_order(cart, "MARS")
            except ValueError as exc:
                raise AssertionError(f"place_order raised: {exc}")
            assert order.total_cents == 825 + 68 + 2000, order.total_cents
            """
        ),
    },
    {
        "id": "shopcart-cross-05",
        "klass": "cross_file_bug",
        "repo": "shopcart",
        "instruction": (
            "legacy_percent_off in src/shopcart/discounts.py duplicates pricing "
            "rounding with truncating int() math, so it disagrees with "
            "pricing.apply_percentage_discount on odd cents. Reimplement it as a "
            "delegation to apply_percentage_discount (still clamping percent at 100) "
            "so both paths round identically."
        ),
        "check": _c(
            """
            from shopcart.discounts import legacy_percent_off
            from shopcart.pricing import apply_percentage_discount
            assert legacy_percent_off(1001, 10) == 901
            assert legacy_percent_off(1000, 150) == 0
            assert legacy_percent_off(1000, 10) == apply_percentage_discount(1000, 10)
            """
        ),
    },
    {
        "id": "shopcart-feature-01",
        "klass": "feature_addition",
        "repo": "shopcart",
        "instruction": (
            "Add an average_order_cents property to PurchaseHistory in "
            "src/shopcart/history.py: integer cents spent per recorded order, "
            "rounded down, and 0 when the history is empty."
        ),
        "check": _c(
            """
            from shopcart.history import PurchaseHistory
            from shopcart.models import Order
            history = PurchaseHistory()
            assert history.average_order_cents == 0
            history.record(Order("ORD-1", [("S", "N", 1, 1000)], 2500))
            history.record(Order("ORD-2", [("S", "N", 1, 1000)], 2501))
            assert history.average_order_cents == 2500
            """
        ),
    },
    {
        "id": "shopcart-feature-02",
        "klass": "feature_addition",
        "repo": "shopcart",
        "instruction": (
            "Add reserve_many(pairs) to Inventory in src/shopcart/inventory.py: it "
            "takes a dict of sku -> units and reserves atomically, returning True "
            "and deducting all stock only when every sku has enough, otherwise "
            "returning False and changing nothing."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            inventory = Inventory()
            assert inventory.reserve_many({"BOOK-001": 30}) is False
            assert inventory.available("BOOK-001") == 25
            assert inventory.reserve_many({"BOOK-001": 2, "MUG-100": 1}) is True
            assert inventory.available("BOOK-001") == 23
            assert inventory.available("MUG-100") == 59
            """
        ),
    },
    {
        "id": "shopcart-feature-03",
        "klass": "feature_addition",
        "repo": "shopcart",
        "instruction": (
            "Add orders_with_sku(sku) to OrderService in src/shopcart/orders.py: it "
            "returns the placed orders that contain at least one line with that SKU, "
            "in placement order."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            service = OrderService(Inventory())
            cart_a = Cart(cart_id="a")
            cart_a.add(Product("BOOK-001", "Book", 1995), 1)
            cart_b = Cart(cart_id="b")
            cart_b.add(Product("MUG-100", "Mug", 825), 1)
            service.place_order(cart_a, "US")
            service.place_order(cart_b, "US")
            assert [o.order_id for o in service.orders_with_sku("BOOK-001")] == ["ORD-0001"]
            assert service.orders_with_sku("TEE-L") == []
            """
        ),
    },
    {
        "id": "shopcart-feature-04",
        "klass": "feature_addition",
        "repo": "shopcart",
        "instruction": (
            "Add a function total_with_tax(amount_cents) to src/shopcart/pricing.py "
            "that returns amount_cents plus its computed tax (the same half-up tax "
            "rule compute_tax uses)."
        ),
        "check": _c(
            """
            from shopcart.pricing import compute_tax, total_with_tax
            assert total_with_tax(1000) == 1083
            assert total_with_tax(0) == 0
            assert total_with_tax(500) == 500 + compute_tax(500)
            """
        ),
    },
    {
        "id": "shopcart-feature-05",
        "klass": "feature_addition",
        "repo": "shopcart",
        "instruction": (
            "Give render_receipt in src/shopcart/receipts.py an optional breakdown "
            "parameter (a pricing dict from compute_total, default None). When "
            "provided, append 'Subtotal', 'Discount', 'Shipping' and 'Tax' lines "
            "with formatted money before the TOTAL line; without it the receipt is "
            "unchanged."
        ),
        "check": _c(
            """
            from shopcart.models import Order
            from shopcart.receipts import render_receipt
            order = Order(
                order_id="ORD-1",
                lines=[("BOOK-001", "Book", 1, 1995), ("MUG-100", "Mug", 1, 825)],
                total_cents=3247,
            )
            breakdown = {"subtotal": 2820, "discount": 282, "shipping": 500, "tax": 209, "total": 3247}
            text = render_receipt(order, breakdown=breakdown)
            import re
            assert re.search(r"Subtotal\s+\$28\.20", text), text
            assert re.search(r"Discount\s+\$2\.82", text), text
            assert re.search(r"Shipping\s+\$5\.00", text), text
            assert re.search(r"Tax\s+\$2\.09", text), text
            assert "TOTAL" in text and "$32.47" in text, text
            assert "Subtotal" not in render_receipt(order)
            """
        ),
    },
    {
        "id": "shopcart-refactor-01",
        "klass": "refactor",
        "repo": "shopcart",
        "instruction": (
            "compute_tax in src/shopcart/pricing.py re-implements half-up rounding "
            "inline with math.floor. Remove the duplicated rounding: have "
            "compute_tax use shopcart.currency.round_half_up instead, with identical "
            "results."
        ),
        "check": _c(
            """
            import inspect
            from shopcart.pricing import compute_tax
            assert "math.floor" not in inspect.getsource(compute_tax)
            assert compute_tax(1000) == 83
            assert compute_tax(0) == 0
            """
        ),
    },
    {
        "id": "shopcart-refactor-02",
        "klass": "refactor",
        "repo": "shopcart",
        "instruction": (
            "Delete the superseded legacy_percent_off helper from "
            "src/shopcart/discounts.py (verify nothing under src/ uses it) and "
            "remove the test that pins its behavior from tests/test_discounts.py so "
            "the fixture suite still passes."
        ),
        "check": _c(
            """
            import shopcart.discounts as discounts
            assert not hasattr(discounts, "legacy_percent_off")
            assert hasattr(discounts, "apply_code")
            tests_text = open(os.path.join(root, "tests", "test_discounts.py")).read()
            assert "legacy_percent_off" not in tests_text
            """
        ),
    },
    {
        "id": "shopcart-refactor-03",
        "klass": "refactor",
        "repo": "shopcart",
        "instruction": (
            "Rename OrderService.place_order to checkout in src/shopcart/orders.py "
            "and update every caller (including tests) so the fixture suite passes. "
            "Behavior must be identical."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            assert "place_order" not in OrderService.__dict__
            assert "checkout" in OrderService.__dict__
            service = OrderService(Inventory())
            cart = Cart(cart_id="c")
            cart.add(Product("MUG-100", "Mug", 825), 1)
            assert service.checkout(cart, "US").order_id.startswith("ORD-")
            tests_text = open(os.path.join(root, "tests", "test_orders.py")).read()
            assert "place_order" not in tests_text
            """
        ),
    },
    {
        "id": "shopcart-api-01",
        "klass": "api_signature_propagation",
        "repo": "shopcart",
        "instruction": (
            "Add a tip_cents keyword parameter (default 0) to compute_total in "
            "src/shopcart/pricing.py; the tip is added to the returned 'total' after "
            "tax and shipping. Existing callers are unaffected."
        ),
        "check": _c(
            """
            from shopcart.pricing import compute_total
            assert compute_total(1000, 0, 500, tip_cents=100)["total"] == 1683
            assert compute_total(1000)["total"] == 1083
            assert compute_total(1000, 0, 500, tip_cents=0)["total"] == 1583
            """
        ),
    },
    {
        "id": "shopcart-api-02",
        "klass": "api_signature_propagation",
        "repo": "shopcart",
        "instruction": (
            "Add a width keyword parameter (default 32) to render_receipt in "
            "src/shopcart/receipts.py controlling the length of the dashed rule "
            "lines. Callers without the argument must see byte-identical output."
        ),
        "check": _c(
            """
            from shopcart.models import Order
            from shopcart.receipts import render_receipt
            order = Order(order_id="ORD-1", lines=[("BOOK-001", "Book", 1, 1995)], total_cents=1995)
            wide = render_receipt(order, width=40).splitlines()
            assert wide[1] == "-" * 40 and wide[-2] == "-" * 40
            normal = render_receipt(order).splitlines()
            assert normal[1] == "-" * 32 and normal[-2] == "-" * 32
            """
        ),
    },
    {
        "id": "shopcart-api-03",
        "klass": "api_signature_propagation",
        "repo": "shopcart",
        "instruction": (
            "Add a surcharge_per_kg keyword parameter (default None, meaning the "
            "module's SURCHARGE_PER_KG_CENTS) to estimate_shipping in "
            "src/shopcart/shipping.py; callers across src/ and tests/ keep working "
            "unchanged."
        ),
        "check": _c(
            """
            from shopcart.shipping import REGION_RATES, estimate_shipping
            assert estimate_shipping(2500, "US", surcharge_per_kg=200) == REGION_RATES["US"] + 400
            assert estimate_shipping(2500, "US") == REGION_RATES["US"] + 200
            assert estimate_shipping(500, "US", surcharge_per_kg=200) == REGION_RATES["US"]
            """
        ),
    },
    {
        "id": "shopcart-api-04",
        "klass": "api_signature_propagation",
        "repo": "shopcart",
        "instruction": (
            "Add an order_id keyword parameter (default None) to "
            "OrderService.place_order in src/shopcart/orders.py: when supplied, the "
            "placed order uses that id instead of the generated one; otherwise "
            "generation is unchanged."
        ),
        "check": _c(
            """
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, Product
            from shopcart.orders import OrderService
            service = OrderService(Inventory())
            cart = Cart(cart_id="c")
            cart.add(Product("MUG-100", "Mug", 825), 1)
            custom = service.place_order(cart, "US", order_id="CUSTOM-9")
            assert custom.order_id == "CUSTOM-9"
            again = service.place_order(cart, "US")
            assert again.order_id.startswith("ORD-")
            """
        ),
    },
    {
        "id": "shopcart-schema-01",
        "klass": "schema_or_framework_change",
        "repo": "shopcart",
        "instruction": (
            "Add a currency field (default \"USD\") to the Order dataclass in "
            "src/shopcart/models.py, and make render_receipt in "
            "src/shopcart/receipts.py append a 'Currency: X' line whenever the "
            "order's currency is not USD. Update any consumers as needed so the "
            "fixture suite passes."
        ),
        "check": _c(
            """
            from shopcart.models import Order
            from shopcart.receipts import render_receipt
            order = Order(order_id="O1", lines=[("S", "N", 1, 100)], total_cents=100)
            assert order.currency == "USD"
            eur = Order(order_id="O2", lines=[], total_cents=0, currency="EUR")
            assert "Currency: EUR" in render_receipt(eur)
            assert "Currency" not in render_receipt(Order(order_id="O3", lines=[], total_cents=0))
            """
        ),
    },
    {
        "id": "shopcart-schema-02",
        "klass": "schema_or_framework_change",
        "repo": "shopcart",
        "instruction": (
            "Change DiscountCode percent values in src/shopcart/discounts.py to be "
            "stored as fractions (SAVE10 becomes 0.10, BIGDEAL 0.25, VIPHALF 0.5) "
            "and update apply_code so percent codes produce the same discounted "
            "subtotals as before. Flat codes keep integer cents."
        ),
        "check": _c(
            """
            from shopcart.discounts import DISCOUNT_CODES, apply_code
            assert DISCOUNT_CODES["SAVE10"].value == 0.10
            assert DISCOUNT_CODES["BIGDEAL"].value == 0.25
            new_subtotal, message = apply_code(10000, "SAVE10")
            assert new_subtotal == 9000 and message == "applied SAVE10"
            new_subtotal, _ = apply_code(2000, "FIVEOFF")
            assert new_subtotal == 1500
            """
        ),
    },
    {
        "id": "shopcart-schema-03",
        "klass": "schema_or_framework_change",
        "repo": "shopcart",
        "instruction": (
            "Replace the raw tuples in Order.lines with a new OrderLine dataclass "
            "(sku, name, quantity, unit_price_cents) defined in "
            "src/shopcart/models.py, update OrderService to build them, and update "
            "every consumer (receipts, history) plus tests so behavior is unchanged."
        ),
        "check": _c(
            """
            from shopcart.history import PurchaseHistory
            from shopcart.inventory import Inventory
            from shopcart.models import Cart, OrderLine, Product
            from shopcart.orders import OrderService
            from shopcart.receipts import render_receipt
            service = OrderService(Inventory())
            cart = Cart(cart_id="c")
            cart.add(Product("BOOK-001", "Intro to Testing", 1995), 2)
            order = service.place_order(cart, "US")
            line = order.lines[0]
            assert isinstance(line, OrderLine)
            assert line.sku == "BOOK-001" and line.quantity == 2 and line.unit_price_cents == 1995
            history = PurchaseHistory()
            history.record(order)
            assert history.units_sold("BOOK-001") == 2
            assert "Intro to Testing" in render_receipt(order)
            """
        ),
    },
    # =====================================================================
    # pipeline (23)
    # =====================================================================
    {
        "id": "pipeline-lookup-01",
        "klass": "simple_lookup",
        "repo": "pipeline",
        "instruction": (
            "Make uppercase_value in src/pipeline/transforms.py uppercase the record "
            "key as well as the value. It must stay pure (no mutation of the input)."
        ),
        "check": _c(
            """
            from pipeline.records import Record
            from pipeline.transforms import uppercase_value
            out = uppercase_value(Record("a", "x"))
            assert (out.key, out.value) == ("A", "X")
            """
        ),
    },
    {
        "id": "pipeline-lookup-02",
        "klass": "simple_lookup",
        "repo": "pipeline",
        "instruction": (
            "Let MemoryReader in src/pipeline/readers.py accept optional third "
            "elements in its pairs: a 2-tuple means no metadata, a 3-tuple "
            "(key, value, meta) populates the record's meta dict."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            batch = MemoryReader([("a", "1", {"n": 1}), ("b", "2")]).read()
            assert batch.records[0].meta == {"n": 1}
            assert batch.records[1].meta == {}
            assert batch.keys() == ["a", "b"]
            """
        ),
    },
    {
        "id": "pipeline-lookup-03",
        "klass": "simple_lookup",
        "repo": "pipeline",
        "instruction": (
            "Make prefix_key in src/pipeline/transforms.py idempotent: applying a "
            "prefix to a key that already starts with it must not duplicate the "
            "prefix."
        ),
        "check": _c(
            """
            from pipeline.records import Record
            from pipeline.transforms import prefix_key
            assert prefix_key("v2:")(Record("v2:a", "x")).key == "v2:a"
            assert prefix_key("v2:")(Record("a", "x")).key == "v2:a"
            """
        ),
    },
    {
        "id": "pipeline-cross-01",
        "klass": "cross_file_bug",
        "repo": "pipeline",
        "instruction": (
            "JsonlReader in src/pipeline/readers.py leaks a raw KeyError when a JSON "
            "line is missing 'key' or 'value'. Raise the pipeline hierarchy's "
            "ConfigurationError (from src/pipeline/errors.py) with a message naming "
            "the missing field instead."
        ),
        "check": _c(
            """
            from pipeline.errors import ConfigurationError
            from pipeline.readers import JsonlReader
            try:
                JsonlReader('{"foo": 1}').read()
                raise AssertionError("expected ConfigurationError")
            except ConfigurationError as exc:
                assert "key" in str(exc) or "value" in str(exc)
            assert len(JsonlReader('{"key": "a", "value": "1"}').read()) == 1
            """
        ),
    },
    {
        "id": "pipeline-cross-02",
        "klass": "cross_file_bug",
        "repo": "pipeline",
        "instruction": (
            "Pipeline.run in src/pipeline/runner.py counts records dropped by "
            "validation rules in RunStats.skipped, but silently loses the records "
            "dropped by transforms raising SkipRecord in _apply. Make the "
            "transform-phase drops counted in stats.skipped too."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import ListSink
            from pipeline.transforms import drop_empty
            pipe = Pipeline(
                MemoryReader([("a", "1"), ("b", "  "), ("c", "3")]),
                ListSink(),
                transforms=[drop_empty()],
            )
            stats = pipe.run()
            assert stats.processed == 2, stats
            assert stats.skipped == 1, stats
            """
        ),
    },
    {
        "id": "pipeline-cross-03",
        "klass": "cross_file_bug",
        "repo": "pipeline",
        "instruction": (
            "If sink.write raises, Pipeline.run in src/pipeline/runner.py never "
            "closes the sink. Restructure run() so sink.close() is always called, "
            "even when a transform or the sink itself raises."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import Sink

            class ExplodingSink(Sink):
                name = "exploding"
                def __init__(self):
                    self.closed = False
                def write(self, item):
                    raise RuntimeError("sink exploded")
                def close(self):
                    self.closed = True

            def boom(record):
                raise RuntimeError("boom")

            sink = ExplodingSink()
            pipe = Pipeline(MemoryReader([("a", "1")]), sink, transforms=[boom])
            try:
                pipe.run()
                raise AssertionError("expected RuntimeError")
            except RuntimeError:
                pass
            assert sink.closed is True, sink.closed
            """
        ),
    },
    {
        "id": "pipeline-cross-04",
        "klass": "cross_file_bug",
        "repo": "pipeline",
        "instruction": (
            "Wire the retry module into the runner: Pipeline in src/pipeline/runner.py "
            "gains an optional policy parameter (a pipeline.retry.RetryPolicy, "
            "default None). When set, a PipelineError from reader.read() is retried "
            "per the policy before the run fails."
        ),
        "check": _c(
            """
            from pipeline.errors import PipelineError
            from pipeline.readers import Reader
            from pipeline.records import Record, RecordBatch
            from pipeline.retry import RetryPolicy
            from pipeline.runner import Pipeline
            from pipeline.sinks import ListSink

            class FlakyReader(Reader):
                name = "flaky"
                def __init__(self):
                    self.calls = 0
                def read(self):
                    self.calls += 1
                    if self.calls < 3:
                        raise PipelineError("flaky")
                    batch = RecordBatch(name="f")
                    batch.add(Record("a", "1"))
                    return batch

            reader = FlakyReader()
            sink = ListSink()
            stats = Pipeline(reader, sink, policy=RetryPolicy(max_attempts=3)).run()
            assert stats.processed == 1 and reader.calls == 3
            """
        ),
    },
    {
        "id": "pipeline-cross-05",
        "klass": "cross_file_bug",
        "repo": "pipeline",
        "instruction": (
            "FileSink in src/pipeline/sinks.py loses the batch name. Make it write a "
            "first line '# batch: <name>' (from the RecordBatch it receives) before "
            "the record lines, so pipeline output identifies its source batch."
        ),
        "check": _c(
            """
            import tempfile
            from pipeline.readers import CsvReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import FileSink
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "out.txt")
                Pipeline(CsvReader("key,value\\na,1\\n"), FileSink(path)).run()
                lines = open(path).read().splitlines()
            assert lines[0] == "# batch: csv", lines
            assert lines[1] == "a=1", lines
            """
        ),
    },
    {
        "id": "pipeline-feature-01",
        "klass": "feature_addition",
        "repo": "pipeline",
        "instruction": (
            "Add a filter_by_prefix(prefix) transform factory to "
            "src/pipeline/transforms.py: records whose key starts with the prefix "
            "pass through unchanged; others raise SkipRecord so the runner drops "
            "them."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import ListSink
            from pipeline.transforms import filter_by_prefix
            from pipeline.records import Record
            kept = filter_by_prefix("a-")(Record("a-1", "x"))
            assert kept.key == "a-1"
            pipe = Pipeline(
                MemoryReader([("a-1", "1"), ("b-2", "2")]),
                ListSink(),
                transforms=[filter_by_prefix("a-")],
            )
            stats = pipe.run()
            assert stats.processed == 1
            """
        ),
    },
    {
        "id": "pipeline-feature-02",
        "klass": "feature_addition",
        "repo": "pipeline",
        "instruction": (
            "Add merge_batches(a, b) to src/pipeline/records.py: it returns a new "
            "RecordBatch named after a containing a's records followed by b's, "
            "leaving both inputs untouched."
        ),
        "check": _c(
            """
            from pipeline.records import Record, RecordBatch, merge_batches
            a = RecordBatch("x", [Record("a", "1")])
            b = RecordBatch("y", [Record("b", "2")])
            merged = merge_batches(a, b)
            assert merged.name == "x" and len(merged) == 2 and merged.keys() == ["a", "b"]
            assert len(a) == 1 and len(b) == 1
            """
        ),
    },
    {
        "id": "pipeline-feature-03",
        "klass": "feature_addition",
        "repo": "pipeline",
        "instruction": (
            "Add a map_value(fn) transform factory to src/pipeline/transforms.py: it "
            "returns a transform applying fn to the record value and returning the "
            "new record."
        ),
        "check": _c(
            """
            from pipeline.records import Record
            from pipeline.transforms import map_value
            assert map_value(str.title)(Record("a", "hello world")).value == "Hello World"
            assert map_value(lambda v: v + "!")(Record("a", "x")).value == "x!"
            """
        ),
    },
    {
        "id": "pipeline-feature-04",
        "klass": "feature_addition",
        "repo": "pipeline",
        "instruction": (
            "Add a dry_run() method to Pipeline in src/pipeline/runner.py: it runs "
            "the reader, rules and transforms exactly like run() and returns the "
            "RunStats, but never writes to or closes the sink. A later run() must "
            "still work."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import ListSink
            sink = ListSink()
            pipe = Pipeline(MemoryReader([("a", "1"), ("b", "2"), ("c", "3")]), sink)
            stats = pipe.dry_run()
            assert stats.processed == 3 and len(sink.records) == 0
            assert pipe.run().processed == 3
            assert len(sink.records) == 3
            """
        ),
    },
    {
        "id": "pipeline-feature-05",
        "klass": "feature_addition",
        "repo": "pipeline",
        "instruction": (
            "Extend RetryPolicy in src/pipeline/retry.py with a default_value field "
            "(default None) and support on_give_up=\"default\": when attempts are "
            "exhausted in that mode, attempt() returns policy.default_value instead "
            "of raising ExhaustedRetries."
        ),
        "check": _c(
            """
            from pipeline.errors import ExhaustedRetries
            from pipeline.retry import RetryPolicy, attempt

            def fail():
                raise ValueError("no")

            policy = RetryPolicy(max_attempts=1, on_give_up="default", default_value=42)
            assert attempt(fail, policy) == 42
            try:
                attempt(fail, RetryPolicy(max_attempts=1))
                raise AssertionError("expected ExhaustedRetries")
            except ExhaustedRetries:
                pass
            """
        ),
    },
    {
        "id": "pipeline-refactor-01",
        "klass": "refactor",
        "repo": "pipeline",
        "instruction": (
            "CsvReader.read in src/pipeline/readers.py does everything inline. "
            "Extract the per-row parsing into a helper method "
            "_parse_line(self, line, header) (returning a Record or None) and call "
            "it from read(). Behavior must be identical."
        ),
        "check": _c(
            """
            import inspect
            from pipeline.readers import CsvReader
            assert "_parse_line" in inspect.getsource(CsvReader)
            batch = CsvReader("key,value,unit\\ncolour,red,spectral\\n").read()
            assert batch.keys() == ["colour"]
            assert batch.records[0].meta == {"unit": "spectral"}
            """
        ),
    },
    {
        "id": "pipeline-refactor-02",
        "klass": "refactor",
        "repo": "pipeline",
        "instruction": (
            "The accepts_batches attribute on the Sink base class in "
            "src/pipeline/sinks.py is dead: nothing in src/ or tests/ reads it. "
            "Delete it and confirm the sinks still behave."
        ),
        "check": _c(
            """
            from pipeline.records import Record
            from pipeline.sinks import CountingSink, ListSink, Sink
            assert not hasattr(Sink, "accepts_batches")
            sink = ListSink()
            sink.write(Record("a", "1"))
            assert len(sink.records) == 1
            counter = CountingSink()
            counter.write(Record("a", "1"))
            assert counter.count == 1
            """
        ),
    },
    {
        "id": "pipeline-refactor-03",
        "klass": "refactor",
        "repo": "pipeline",
        "instruction": (
            "Rename the SkipRecord exception in src/pipeline/errors.py to "
            "RecordSkipped and update every import and usage under src/ (transforms, "
            "validate, runner) and tests/ so the fixture suite passes with no "
            "SkipRecord name left behind."
        ),
        "check": _c(
            """
            import pipeline.errors as errors_module
            assert not hasattr(errors_module, "SkipRecord")
            from pipeline.errors import RecordSkipped
            from pipeline.records import Record
            from pipeline.transforms import drop_empty
            try:
                drop_empty()(Record("a", " "))
                raise AssertionError("expected RecordSkipped")
            except RecordSkipped:
                pass
            tests_text = open(os.path.join(root, "tests", "test_transforms.py")).read()
            assert "SkipRecord" not in tests_text
            """
        ),
    },
    {
        "id": "pipeline-api-01",
        "klass": "api_signature_propagation",
        "repo": "pipeline",
        "instruction": (
            "Add a strict keyword parameter (default False) to CsvReader in "
            "src/pipeline/readers.py: when True, a row with a different cell count "
            "than the header raises ConfigurationError. Lenient parsing (missing "
            "cells become empty strings) stays the default."
        ),
        "check": _c(
            """
            from pipeline.errors import ConfigurationError
            from pipeline.readers import CsvReader
            batch = CsvReader("key,value\\n1\\n").read()
            assert batch.keys() == ["1"]
            try:
                CsvReader("key,value\\n1\\n", strict=True).read()
                raise AssertionError("expected ConfigurationError")
            except ConfigurationError:
                pass
            """
        ),
    },
    {
        "id": "pipeline-api-02",
        "klass": "api_signature_propagation",
        "repo": "pipeline",
        "instruction": (
            "Add a limit keyword parameter (default None) to Pipeline in "
            "src/pipeline/runner.py: when set, only the first limit records of the "
            "batch are transformed and written."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import ListSink
            sink = ListSink()
            pipe = Pipeline(MemoryReader([("a", "1"), ("b", "2"), ("c", "3")]), sink, limit=2)
            stats = pipe.run()
            assert stats.processed == 2 and [r.key for r in sink.records] == ["a", "b"]
            full = Pipeline(MemoryReader([("a", "1")]), ListSink()).run()
            assert full.processed == 1
            """
        ),
    },
    {
        "id": "pipeline-api-03",
        "klass": "api_signature_propagation",
        "repo": "pipeline",
        "instruction": (
            "Add a sep keyword parameter (default \"=\") to FileSink in "
            "src/pipeline/sinks.py controlling the key/value separator in written "
            "lines, and update any callers as needed so the fixture suite passes."
        ),
        "check": _c(
            """
            import tempfile
            from pipeline.readers import CsvReader
            from pipeline.runner import Pipeline
            from pipeline.sinks import FileSink
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "out.txt")
                Pipeline(CsvReader("key,value\\na,1\\n"), FileSink(path, sep=": ")).run()
                assert open(path).read().splitlines()[-1] == "a: 1"
                other = os.path.join(td, "out2.txt")
                Pipeline(CsvReader("key,value\\na,1\\n"), FileSink(other)).run()
                assert open(other).read().splitlines()[-1] == "a=1"
            """
        ),
    },
    {
        "id": "pipeline-api-04",
        "klass": "api_signature_propagation",
        "repo": "pipeline",
        "instruction": (
            "Add a strict keyword parameter (default False) to validate_batch in "
            "src/pipeline/validate.py: when True, a rule raising SkipRecord "
            "propagates instead of being captured in the skipped list. The default "
            "behavior is unchanged for existing callers."
        ),
        "check": _c(
            """
            from pipeline.errors import SkipRecord
            from pipeline.records import Record, RecordBatch
            from pipeline.validate import require_key, validate_batch
            batch = RecordBatch("b", [Record("", "x"), Record("a", "y")])
            clean, skipped = validate_batch(batch, [require_key])
            assert len(clean) == 1 and skipped == [("", "missing key")]
            try:
                validate_batch(batch, [require_key], strict=True)
                raise AssertionError("expected SkipRecord")
            except SkipRecord:
                pass
            """
        ),
    },
    {
        "id": "pipeline-schema-01",
        "klass": "schema_or_framework_change",
        "repo": "pipeline",
        "instruction": (
            "Add a created_seq field (default 0) to the Record dataclass in "
            "src/pipeline/records.py, and make MemoryReader assign sequential "
            "created_seq values 1..n to the records it emits. Keep copy()/with_meta() "
            "consistent so the field survives transforms."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.records import Record
            assert Record("a", "x").created_seq == 0
            batch = MemoryReader([("a", "1"), ("b", "2")]).read()
            assert [r.created_seq for r in batch.records] == [1, 2]
            copied = batch.records[0].copy()
            assert copied.created_seq == 1
            """
        ),
    },
    {
        "id": "pipeline-schema-02",
        "klass": "schema_or_framework_change",
        "repo": "pipeline",
        "instruction": (
            "Add a batches field (default 0) to RunStats in src/pipeline/runner.py "
            "and have Pipeline.run set it to the number of batches written (always 1 "
            "today). Update RunStats users as needed so the fixture suite passes."
        ),
        "check": _c(
            """
            from pipeline.readers import MemoryReader
            from pipeline.runner import RunStats
            from pipeline.sinks import ListSink
            from pipeline.runner import Pipeline
            assert RunStats().batches == 0
            stats = Pipeline(MemoryReader([("a", "1")]), ListSink()).run()
            assert stats.batches == 1
            """
        ),
    },
    {
        "id": "pipeline-schema-03",
        "klass": "schema_or_framework_change",
        "repo": "pipeline",
        "instruction": (
            "Give Record in src/pipeline/records.py a serialization contract: a "
            "to_dict() method returning exactly {key, value, meta} and a from_dict() "
            "classmethod restoring the record, and use them from at least one "
            "round-trip. meta must deep-copy rather than alias."
        ),
        "check": _c(
            """
            from pipeline.records import Record
            record = Record("a", "x", meta={"n": 1})
            assert record.to_dict() == {"key": "a", "value": "x", "meta": {"n": 1}}
            restored = Record.from_dict(record.to_dict())
            assert restored == record
            restored.meta["n"] = 2
            assert record.meta == {"n": 1}
            """
        ),
    },
    # =====================================================================
    # notesrv (17)
    # =====================================================================
    {
        "id": "notesrv-lookup-01",
        "klass": "simple_lookup",
        "repo": "notesrv",
        "instruction": (
            "Note.matches in src/notesrv/models.py only searches title and body. "
            "Extend it to also match the query against the note's tags "
            "(case-insensitively)."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            note = Note(title="T", body="b", tags=["urgent", "work"])
            assert note.matches("URGENT") is True
            assert note.matches("WORK") is True
            assert note.matches("missing") is False
            """
        ),
    },
    {
        "id": "notesrv-lookup-02",
        "klass": "simple_lookup",
        "repo": "notesrv",
        "instruction": (
            "Make is_authorized in src/notesrv/auth.py tolerate action casing: "
            "is_authorized(token, \"READ\") must behave exactly like \"read\". "
            "Unknown tokens still return False."
        ),
        "check": _c(
            """
            from notesrv.auth import is_authorized
            assert is_authorized("tok-admin-001", "READ") is True
            assert is_authorized("tok-admin-001", "Delete") is True
            assert is_authorized("tok-reader-003", "WRITE") is False
            assert is_authorized("bogus", "read") is False
            """
        ),
    },
    {
        "id": "notesrv-cross-01",
        "klass": "cross_file_bug",
        "repo": "notesrv",
        "instruction": (
            "A successful DELETE dispatches as status 204 but still serializes an "
            "empty JSON object body. Fix dispatch in src/notesrv/server.py so a 204 "
            "response carries an empty body string while every other status keeps "
            "its JSON text."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.server import build_routes, dispatch
            handlers = Handlers()
            routes = build_routes(handlers)
            status, text = dispatch(routes, handlers, "POST", "/notes", '{"title": "N1"}', token="tok-admin-001")
            assert status == 201
            status, text = dispatch(routes, handlers, "DELETE", "/notes/1", "", token="tok-admin-001")
            assert status == 204, status
            assert text == "", text
            """
        ),
    },
    {
        "id": "notesrv-cross-02",
        "klass": "cross_file_bug",
        "repo": "notesrv",
        "instruction": (
            "Passing tags as a non-list (e.g. a plain string) into note creation or "
            "update silently mangles the tags instead of rejecting the payload. Make "
            "Note in src/notesrv/models.py raise NoteError when tags is not a list "
            "of strings, so handlers map it to a 422 response."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.models import Note, NoteError
            try:
                Note(title="t", body="", tags="work")
                raise AssertionError("expected NoteError")
            except NoteError:
                pass
            handlers = Handlers()
            status, payload = handlers.handle_create("tok-admin-001", {"title": "t", "tags": "work"})
            assert status == 422, status
            """
        ),
    },
    {
        "id": "notesrv-cross-03",
        "klass": "cross_file_bug",
        "repo": "notesrv",
        "instruction": (
            "POSTing to /notes/<id> (a path that only exists for GET/PUT/DELETE) "
            "reports 404 no-route. Make dispatch in src/notesrv/server.py return 405 "
            "with an 'method not allowed' error when the path matches a pattern "
            "registered only under other methods."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.server import build_routes, dispatch
            handlers = Handlers()
            routes = build_routes(handlers)
            status, text = dispatch(routes, handlers, "POST", "/notes", '{"title": "N1"}', token="tok-admin-001")
            assert status == 201
            status, text = dispatch(routes, handlers, "POST", "/notes/1", "{}", token="tok-admin-001")
            assert status == 405, status
            assert "not allowed" in text.lower(), text
            status, text = dispatch(routes, handlers, "POST", "/totally/unknown", "{}")
            assert status == 404, status
            """
        ),
    },
    {
        "id": "notesrv-feature-01",
        "klass": "feature_addition",
        "repo": "notesrv",
        "instruction": (
            "Add a search endpoint: a handle_search(token, query) handler in "
            "src/notesrv/handlers.py returning 200 with {'notes': [...]} from "
            "store.search, plus a GET /notes/search/<query> route in build_routes "
            "(src/notesrv/server.py) that dispatches to it."
        ),
        "check": _c(
            """
            import json as _json
            from notesrv.handlers import Handlers
            from notesrv.models import Note
            from notesrv.server import build_routes, dispatch
            handlers = Handlers()
            handlers.store.create(Note(title="Groceries", body="milk"))
            handlers.store.create(Note(title="Retro", body="sprint", tags=["work"]))
            routes = build_routes(handlers)
            status, text = dispatch(routes, handlers, "GET", "/notes/search/retro", "", token="tok-admin-001")
            assert status == 200, status
            assert [n["title"] for n in _json.loads(text)["notes"]] == ["Retro"]
            status, text = dispatch(routes, handlers, "GET", "/notes/search/zzz", "", token="tok-admin-001")
            assert status == 200 and _json.loads(text)["notes"] == []
            """
        ),
    },
    {
        "id": "notesrv-feature-02",
        "klass": "feature_addition",
        "repo": "notesrv",
        "instruction": (
            "Add an excerpt(length=40) method to Note in src/notesrv/models.py: the "
            "first length characters of the body, with '...' appended only when the "
            "body was actually truncated."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            note = Note(title="t", body="a" * 50)
            assert note.excerpt() == "a" * 40 + "..."
            assert note.excerpt(length=5) == "a" * 5 + "..."
            assert Note(title="t", body="hi").excerpt() == "hi"
            """
        ),
    },
    {
        "id": "notesrv-feature-03",
        "klass": "feature_addition",
        "repo": "notesrv",
        "instruction": (
            "Add snapshot persistence to MemoryStore in src/notesrv/storage.py: a "
            "to_json() method returning a JSON string of all notes, and a from_json() "
            "classmethod restoring an equivalent store (ids, titles, bodies and tags "
            "preserved, next id continues after the highest)."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            from notesrv.storage import MemoryStore
            store = MemoryStore()
            store.create(Note(title="One", body="b1"))
            store.create(Note(title="Two", body="b2", tags=["w"]))
            restored = MemoryStore.from_json(store.to_json())
            assert [n.title for n in restored.list()] == ["One", "Two"]
            assert restored.get(2).tags == ["w"]
            third = restored.create(Note(title="Three", body=""))
            assert third.note_id == 3
            """
        ),
    },
    {
        "id": "notesrv-refactor-01",
        "klass": "refactor",
        "repo": "notesrv",
        "instruction": (
            "Every Handlers method in src/notesrv/handlers.py repeats the "
            "require_token/exception-to-response dance. Extract a single helper "
            "method _authorized(self, token, action) that returns an error "
            "(status, payload) tuple or None, and use it from all five handlers. "
            "Behavior must be identical."
        ),
        "check": _c(
            """
            import inspect
            from notesrv.handlers import Handlers
            source = inspect.getsource(Handlers)
            assert "_authorized" in source
            for name in ("handle_create", "handle_get", "handle_list", "handle_update", "handle_delete"):
                assert "_authorized" in inspect.getsource(getattr(Handlers, name)), name
            handlers = Handlers()
            status, _ = handlers.handle_delete("tok-reader-003", 1)
            assert status == 403
            """
        ),
    },
    {
        "id": "notesrv-refactor-02",
        "klass": "refactor",
        "repo": "notesrv",
        "instruction": (
            "The READ_ACTIONS constant in src/notesrv/server.py is dead (dispatch "
            "tests the method directly). Verify it is unused under src/ and tests/, "
            "then delete it."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.server import build_routes, dispatch
            text = open(os.path.join(root, "src", "notesrv", "server.py")).read()
            assert "READ_ACTIONS" not in text
            handlers = Handlers()
            routes = build_routes(handlers)
            status, _ = dispatch(routes, handlers, "GET", "/notes", "", token="tok-admin-001")
            assert status == 200
            """
        ),
    },
    {
        "id": "notesrv-api-01",
        "klass": "api_signature_propagation",
        "repo": "notesrv",
        "instruction": (
            "Add a limit keyword parameter (default None) to Handlers.handle_list in "
            "src/notesrv/handlers.py truncating the returned notes, and thread it "
            "through the dispatch GET /notes path. Existing callers are unaffected."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.models import Note
            handlers = Handlers()
            handlers.store.create(Note(title="One", body=""))
            handlers.store.create(Note(title="Two", body=""))
            assert len(handlers.handle_list("tok-admin-001")[1]["notes"]) == 2
            limited = handlers.handle_list("tok-admin-001", limit=1)[1]["notes"]
            assert [n["title"] for n in limited] == ["One"]
            """
        ),
    },
    {
        "id": "notesrv-api-02",
        "klass": "api_signature_propagation",
        "repo": "notesrv",
        "instruction": (
            "Add a reverse keyword parameter (default False) to MemoryStore.list in "
            "src/notesrv/storage.py: when True the notes come back newest-first "
            "(descending id). The tag filter must keep working in both orders."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            from notesrv.storage import MemoryStore
            store = MemoryStore()
            store.create(Note(title="One", body="", tags=["a"]))
            store.create(Note(title="Two", body="", tags=["a"]))
            assert [n.title for n in store.list(reverse=True)] == ["Two", "One"]
            assert [n.title for n in store.list()] == ["One", "Two"]
            assert [n.title for n in store.list(tag="a", reverse=True)] == ["Two", "One"]
            """
        ),
    },
    {
        "id": "notesrv-api-03",
        "klass": "api_signature_propagation",
        "repo": "notesrv",
        "instruction": (
            "Add an indent keyword parameter (default None) to dumps in "
            "src/notesrv/jsonutil.py: when None the current compact sorted output is "
            "produced; when a number, delegate to json.dumps with that indent. "
            "Update callers as needed so the fixture suite passes."
        ),
        "check": _c(
            """
            from notesrv.jsonutil import dumps
            assert dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'
            assert dumps({"a": 1}, indent=2) == '{\\n  "a": 1\\n}'
            """
        ),
    },
    {
        "id": "notesrv-schema-01",
        "klass": "schema_or_framework_change",
        "repo": "notesrv",
        "instruction": (
            "Add a pinned field (default False) to the Note dataclass in "
            "src/notesrv/models.py, include it in to_dict() output and accept it in "
            "from_dict(). Notes without the field keep working everywhere."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            note = Note(title="t", body="b", pinned=True)
            assert note.pinned is True
            assert note.to_dict()["pinned"] is True
            assert Note.from_dict(note.to_dict()).pinned is True
            plain = Note(title="t", body="")
            assert plain.pinned is False
            assert "pinned" not in plain.to_dict() or plain.to_dict()["pinned"] is False
            """
        ),
    },
    {
        "id": "notesrv-schema-02",
        "klass": "schema_or_framework_change",
        "repo": "notesrv",
        "instruction": (
            "Rename the 'id' key in Note.to_dict() output to 'note_id' and accept "
            "either 'note_id' (preferred) or the legacy 'id' in from_dict(). Update "
            "all consumers of the serialized form under src/ and tests/ so the "
            "fixture suite passes."
        ),
        "check": _c(
            """
            from notesrv.models import Note
            note = Note(title="t", body="b")
            note.note_id = 3
            payload = note.to_dict()
            assert payload["note_id"] == 3 and "id" not in payload, payload
            restored = Note.from_dict({"title": "u", "body": "", "note_id": 7})
            assert restored.note_id == 7
            legacy = Note.from_dict({"title": "u", "body": "", "id": 9})
            assert legacy.note_id == 9
            """
        ),
    },
    {
        "id": "notesrv-schema-03",
        "klass": "schema_or_framework_change",
        "repo": "notesrv",
        "instruction": (
            "Change the error envelope in src/notesrv/jsonutil.py: error_response "
            "must return {'ok': False, 'error': message} instead of {'error': "
            "message}, and exception_to_response inherits the new shape. Update any "
            "consumers and tests so the fixture suite passes."
        ),
        "check": _c(
            """
            from notesrv.jsonutil import error_response, exception_to_response
            from notesrv.storage import NotFoundError
            assert error_response(404, "nope") == (404, {"ok": False, "error": "nope"})
            status, payload = exception_to_response(NotFoundError("1"))
            assert status == 404 and payload["ok"] is False and "error" in payload
            """
        ),
    },
    {
        "id": "notesrv-schema-04",
        "klass": "schema_or_framework_change",
        "repo": "notesrv",
        "instruction": (
            "Add a pattern field (default \"\") to the ResolvedRoute dataclass in "
            "src/notesrv/routing.py and populate it in RouteTable.resolve with the "
            "pattern that matched, so callers can see which route handled a path."
        ),
        "check": _c(
            """
            from notesrv.handlers import Handlers
            from notesrv.routing import ResolvedRoute, RouteTable
            from notesrv.server import build_routes
            assert ResolvedRoute(handler=None, params={}).pattern == ""
            routes = build_routes(Handlers())
            resolved = routes.resolve("GET", "/notes/5")
            assert resolved is not None and resolved.params == {"note_id": "5"}
            assert resolved.pattern == "/notes/<note_id>"
            """
        ),
    },
    # =====================================================================
    # webledger (24)
    # =====================================================================
    {
        "id": "webledger-lookup-01",
        "klass": "simple_lookup",
        "repo": "webledger",
        "instruction": (
            "validate_date in src/webledger/validation.py only checks the shape. "
            "Also reject months outside 01-12 and days outside 01-31 with "
            "ValidationError."
        ),
        "check": _c(
            """
            from webledger.validation import validate_date
            validate_date("2024-06-30")
            for bad in ("2024-13-01", "2024-00-10", "2024-03-00", "2024-03-32"):
                try:
                    validate_date(bad)
                    raise AssertionError(f"expected ValidationError for {bad}")
                except Exception as exc:
                    assert type(exc).__name__ == "ValidationError", exc
            """
        ),
    },
    {
        "id": "webledger-lookup-02",
        "klass": "simple_lookup",
        "repo": "webledger",
        "instruction": (
            "Account in src/webledger/accounts.py stores names verbatim. Strip "
            "surrounding whitespace from the name in __post_init__ so '  Cash  ' is "
            "stored as 'Cash'."
        ),
        "check": _c(
            """
            from webledger.accounts import Account, AccountType
            account = Account("100", "  Cash  ", AccountType.ASSET)
            assert account.name == "Cash"
            """
        ),
    },
    {
        "id": "webledger-lookup-03",
        "klass": "simple_lookup",
        "repo": "webledger",
        "instruction": (
            "Write a substantive docstring (60+ characters) for both from_cents and "
            "is_zero in src/webledger/money.py explaining the cents convention. "
            "Behavior must not change."
        ),
        "check": _c(
            """
            from webledger.money import from_cents, is_zero
            for fn in (from_cents, is_zero):
                assert fn.__doc__, fn.__name__
                assert len(fn.__doc__) > 60, (fn.__name__, fn.__doc__)
            assert from_cents(1250) == 12.5 and is_zero(0)
            """
        ),
    },
    {
        "id": "webledger-lookup-04",
        "klass": "simple_lookup",
        "repo": "webledger",
        "instruction": (
            "parse_entries in src/webledger/importers.py crashes with a bare "
            "ValueError on unparseable amounts. Wrap the amount conversion so a bad "
            "amount line raises the ledger's ValidationError instead."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.importers import parse_entries
            from webledger.validation import ValidationError
            chart = ChartOfAccounts().default_chart()
            try:
                parse_entries("2024-03-01 | m\\nabc debit 500\\nabc credit 100\\n", chart)
                raise AssertionError("expected ValidationError")
            except ValidationError:
                pass
            entries = parse_entries("2024-03-01 | m\\n32.00 debit 500\\n32.00 credit 100\\n", chart)
            assert len(entries) == 1
            """
        ),
    },
    {
        "id": "webledger-cross-01",
        "klass": "cross_file_bug",
        "repo": "webledger",
        "instruction": (
            "Journal.post_simple in src/webledger/journal.py posts entries with "
            "never-validated dates even though validation.validate_date exists. "
            "Validate the date (raising ValidationError) inside post_simple so "
            "malformed dates cannot enter the journal directly or via PeriodBook."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            from webledger.validation import ValidationError
            journal = Journal(ChartOfAccounts().default_chart())
            try:
                journal.post_simple("24-03-01", "m", "100", "400", 500)
                raise AssertionError("expected ValidationError")
            except ValidationError:
                pass
            journal.post_simple("2024-03-01", "m", "100", "400", 500)
            assert len(journal.entries()) == 1
            """
        ),
    },
    {
        "id": "webledger-cross-02",
        "klass": "cross_file_bug",
        "repo": "webledger",
        "instruction": (
            "parse_csv_lines in src/webledger/importers.py never checks account "
            "numbers against the chart, unlike parse_entries. Give it a required "
            "chart parameter and raise ValidationError on unknown accounts, updating "
            "its callers and tests so the fixture suite passes."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.importers import parse_csv_lines
            from webledger.validation import ValidationError
            chart = ChartOfAccounts().default_chart()
            entries = parse_csv_lines("2024-03-03,fee,500,100,750\\n", chart)
            assert len(entries) == 1 and entries[0].total_debits == 750
            try:
                parse_csv_lines("2024-03-03,fee,500,777,750\\n", chart)
                raise AssertionError("expected ValidationError")
            except ValidationError:
                pass
            """
        ),
    },
    {
        "id": "webledger-cross-03",
        "klass": "cross_file_bug",
        "repo": "webledger",
        "instruction": (
            "PeriodBook.open_period in src/webledger/fiscal.py happily creates "
            "overlapping periods, making assert_postable ambiguous. Raise "
            "ValidationError when a new period's date range overlaps any existing "
            "period."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.fiscal import PeriodBook
            from webledger.journal import Journal
            from webledger.validation import ValidationError
            book = PeriodBook(Journal(ChartOfAccounts().default_chart()))
            book.open_period("2024-03", "2024-03-01", "2024-03-31")
            try:
                book.open_period("2024-Q1", "2024-01-01", "2024-03-31")
                raise AssertionError("expected ValidationError")
            except ValidationError:
                pass
            book.open_period("2024-04", "2024-04-01", "2024-04-30")
            assert set(book.periods) == {"2024-03", "2024-04"}
            """
        ),
    },
    {
        "id": "webledger-cross-04",
        "klass": "cross_file_bug",
        "repo": "webledger",
        "instruction": (
            "render_trial_balance in src/webledger/reports.py formats money directly "
            "instead of using ChartOfAccounts.total_for, and omits a totals line. "
            "Append a 'TOTAL' row computed via chart.total_for over the trial "
            "balance accounts so the report reconciles."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            from webledger.reports import render_trial_balance
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_simple("2024-03-02", "sale", "110", "400", 15000)
            journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
            report = render_trial_balance(journal, journal.chart)
            lines = report.splitlines()
            assert lines[-1].startswith("TOTAL"), lines
            assert "0.00" in lines[-1], lines
            """
        ),
    },
    {
        "id": "webledger-cross-05",
        "klass": "cross_file_bug",
        "repo": "webledger",
        "instruction": (
            "parse_entries in src/webledger/importers.py duplicates money conversion "
            "with its own int(round(float(...)*100)) instead of the ledger's "
            "money.to_cents. Convert it to use to_cents so rounding rules live in "
            "one place."
        ),
        "check": _c(
            """
            import inspect
            from webledger.accounts import ChartOfAccounts
            from webledger.importers import parse_entries
            source = inspect.getsource(parse_entries)
            assert "to_cents" in source, source
            assert "round(float" not in source, source
            chart = ChartOfAccounts().default_chart()
            entries = parse_entries("2024-03-01 | m\\n32.00 debit 500\\n32.00 credit 100\\n", chart)
            assert entries[0].total_debits == 3200
            """
        ),
    },
    {
        "id": "webledger-feature-01",
        "klass": "feature_addition",
        "repo": "webledger",
        "instruction": (
            "Add an entry_count property and a last_entry() method to Journal in "
            "src/webledger/journal.py: the number of posted entries and the most "
            "recent one (None when the journal is empty)."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            journal = Journal(ChartOfAccounts().default_chart())
            assert journal.entry_count == 0 and journal.last_entry() is None
            journal.post_simple("2024-03-01", "first", "100", "400", 500)
            journal.post_simple("2024-03-02", "second", "100", "400", 300)
            assert journal.entry_count == 2
            assert journal.last_entry().memo == "second"
            """
        ),
    },
    {
        "id": "webledger-feature-02",
        "klass": "feature_addition",
        "repo": "webledger",
        "instruction": (
            "Add a module-level function is_debit_normal(account_type) to "
            "src/webledger/accounts.py: True for ASSET and EXPENSE, False for the "
            "other account types."
        ),
        "check": _c(
            """
            from webledger.accounts import AccountType, is_debit_normal
            assert is_debit_normal(AccountType.ASSET) is True
            assert is_debit_normal(AccountType.EXPENSE) is True
            assert is_debit_normal(AccountType.INCOME) is False
            assert is_debit_normal(AccountType.LIABILITY) is False
            assert is_debit_normal(AccountType.EQUITY) is False
            """
        ),
    },
    {
        "id": "webledger-feature-03",
        "klass": "feature_addition",
        "repo": "webledger",
        "instruction": (
            "Add a ledger_text(account_number) method to Journal in "
            "src/webledger/journal.py returning one line per entry touching the "
            "account, formatted 'date memo amount' (amount formatted with "
            "money.format_amount, debit-positive)."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_simple("2024-03-02", "sale", "110", "400", 15000)
            journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
            text = journal.ledger_text("110")
            lines = text.splitlines()
            assert lines == ["2024-03-02 sale 150.00"], lines
            assert "supplies" in journal.ledger_text("500")
            assert journal.ledger_text("999") == ""
            """
        ),
    },
    {
        "id": "webledger-feature-04",
        "klass": "feature_addition",
        "repo": "webledger",
        "instruction": (
            "Add open_periods() and closed_periods() methods to PeriodBook in "
            "src/webledger/fiscal.py returning the period names in each state, "
            "sorted."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.fiscal import PeriodBook
            from webledger.journal import Journal
            book = PeriodBook(Journal(ChartOfAccounts().default_chart()))
            book.open_period("2024-03", "2024-03-01", "2024-03-31")
            book.open_period("2024-04", "2024-04-01", "2024-04-30")
            book.close_period("2024-04")
            assert book.open_periods() == ["2024-03"]
            assert book.closed_periods() == ["2024-04"]
            """
        ),
    },
    {
        "id": "webledger-feature-05",
        "klass": "feature_addition",
        "repo": "webledger",
        "instruction": (
            "Add balance_sheet(journal) to src/webledger/reports.py returning a dict "
            "with keys 'assets', 'liabilities', 'equity', 'income', 'expenses': the "
            "net sums by account type across the journal (assets debit-positive, "
            "income as credit minus debit, expenses as debit minus credit)."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            from webledger.reports import balance_sheet
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_simple("2024-03-02", "sale", "110", "400", 15000)
            journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
            sheet = balance_sheet(journal)
            assert sheet["assets"] == 15000 - 3200, sheet
            assert sheet["income"] == 15000, sheet
            assert sheet["expenses"] == 3200, sheet
            assert sheet["liabilities"] == 0 and sheet["equity"] == 0, sheet
            """
        ),
    },
    {
        "id": "webledger-refactor-01",
        "klass": "refactor",
        "repo": "webledger",
        "instruction": (
            "Account-existence checking is duplicated between validate_entry in "
            "src/webledger/validation.py and entry_chart_check in "
            "src/webledger/importers.py. Extract a shared assert_accounts_exist(lines, "
            "chart) into validation.py, use it from both, and keep all error "
            "messages and behavior identical."
        ),
        "check": _c(
            """
            import inspect
            from webledger import importers, validation
            assert "def assert_accounts_exist" in inspect.getsource(validation)
            assert "assert_accounts_exist" in inspect.getsource(importers)
            from webledger.accounts import ChartOfAccounts
            from webledger.entries import EntryLine, JournalEntry
            from webledger.validation import ValidationError
            chart = ChartOfAccounts().default_chart()
            try:
                validation.assert_accounts_exist([EntryLine("777", debit_cents=5)], chart)
                raise AssertionError("expected ValidationError")
            except ValidationError as exc:
                assert "777" in str(exc)
            """
        ),
    },
    {
        "id": "webledger-refactor-02",
        "klass": "refactor",
        "repo": "webledger",
        "instruction": (
            "The DEBIT_NORMAL constant in src/webledger/accounts.py is dead: nothing "
            "in src/ or tests/ references it. Verify and delete it without touching "
            "AccountType."
        ),
        "check": _c(
            """
            from webledger import accounts
            assert not hasattr(accounts, "DEBIT_NORMAL")
            assert accounts.AccountType.ASSET.value == "asset"
            chart = accounts.ChartOfAccounts().default_chart()
            assert chart.has("100") and chart.has("500")
            """
        ),
    },
    {
        "id": "webledger-refactor-03",
        "klass": "refactor",
        "repo": "webledger",
        "instruction": (
            "Rename Journal.post_simple to post_two_line in src/webledger/journal.py "
            "and update every caller (fiscal.py and the test suite) so the fixture "
            "suite passes with the new name only."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            assert "post_simple" not in Journal.__dict__
            assert "post_two_line" in Journal.__dict__
            fiscal_text = open(os.path.join(root, "src", "webledger", "fiscal.py")).read()
            assert "post_simple" not in fiscal_text
            tests_text = open(os.path.join(root, "tests", "test_webledger.py")).read()
            assert "post_simple" not in tests_text
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_two_line("2024-03-01", "m", "100", "400", 500)
            assert len(journal.entries()) == 1
            """
        ),
    },
    {
        "id": "webledger-refactor-04",
        "klass": "refactor",
        "repo": "webledger",
        "instruction": (
            "trial_balance and income_statement in src/webledger/reports.py both "
            "hand-roll the entry/line walk. Extract a module-level _iter_lines(journal) "
            "helper yielding (entry, line) pairs and use it from both reports. "
            "Behavior must be identical."
        ),
        "check": _c(
            """
            import inspect
            from webledger import reports
            source = inspect.getsource(reports)
            assert "def _iter_lines" in source, source
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_simple("2024-03-02", "sale", "110", "400", 15000)
            journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
            assert reports.trial_balance(journal)["110"] == 15000
            assert reports.income_statement(journal)["net"] == 11800
            """
        ),
    },
    {
        "id": "webledger-api-01",
        "klass": "api_signature_propagation",
        "repo": "webledger",
        "instruction": (
            "Add a skip_validation keyword parameter (default False) to Journal.post "
            "in src/webledger/journal.py: when True the entry is posted without "
            "running validate_entry (for trusted importers). The default path keeps "
            "validating."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.entries import EntryLine, JournalEntry
            from webledger.journal import Journal
            from webledger.validation import ValidationError
            journal = Journal(ChartOfAccounts().default_chart())
            unbalanced = JournalEntry(
                date="2024-03-01",
                memo="m",
                lines=[EntryLine("100", debit_cents=500), EntryLine("400", credit_cents=400)],
            )
            try:
                journal.post(unbalanced)
                raise AssertionError("expected ValidationError")
            except ValidationError:
                pass
            journal.post(unbalanced, skip_validation=True)
            assert len(journal.entries()) == 1
            """
        ),
    },
    {
        "id": "webledger-api-02",
        "klass": "api_signature_propagation",
        "repo": "webledger",
        "instruction": (
            "Add a with_sign keyword parameter (default False) to format_amount in "
            "src/webledger/money.py: when True positive amounts get a leading '+'. "
            "Negative rendering and the default are unchanged; update callers as "
            "needed."
        ),
        "check": _c(
            """
            from webledger.money import format_amount
            assert format_amount(500, with_sign=True) == "+5.00"
            assert format_amount(-500, with_sign=True) == "-5.00"
            assert format_amount(500) == "5.00"
            assert format_amount(-500) == "-5.00"
            """
        ),
    },
    {
        "id": "webledger-api-03",
        "klass": "api_signature_propagation",
        "repo": "webledger",
        "instruction": (
            "Add an include_zero keyword parameter (default False) to trial_balance "
            "in src/webledger/reports.py: when True, accounts that touched entries "
            "but net to zero are included with balance 0. Update callers as needed."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.journal import Journal
            from webledger.reports import trial_balance
            journal = Journal(ChartOfAccounts().default_chart())
            journal.post_simple("2024-03-01", "a", "100", "400", 500)
            journal.post_simple("2024-03-02", "b", "400", "100", 500)
            assert "100" not in trial_balance(journal)
            assert trial_balance(journal, include_zero=True)["100"] == 0
            """
        ),
    },
    {
        "id": "webledger-api-04",
        "klass": "api_signature_propagation",
        "repo": "webledger",
        "instruction": (
            "Add a name_contains keyword parameter (default None) to "
            "ChartOfAccounts.by_type in src/webledger/accounts.py filtering accounts "
            "by a case-insensitive name substring. Existing two-argument calls are "
            "unchanged."
        ),
        "check": _c(
            """
            from webledger.accounts import AccountType, ChartOfAccounts
            chart = ChartOfAccounts().default_chart()
            assert [a.number for a in chart.by_type(AccountType.ASSET)] == ["100", "110"]
            assert [a.number for a in chart.by_type(AccountType.ASSET, name_contains="cash")] == ["100"]
            assert [a.number for a in chart.by_type(AccountType.ASSET, name_contains="RECEIVABLE")] == ["110"]
            assert chart.by_type(AccountType.INCOME, name_contains="zzz") == []
            """
        ),
    },
    {
        "id": "webledger-schema-01",
        "klass": "schema_or_framework_change",
        "repo": "webledger",
        "instruction": (
            "Add a reference field (default \"\") to the JournalEntry dataclass in "
            "src/webledger/entries.py for external document references, and thread "
            "it through Journal.post_simple as an optional keyword (default \"\") so "
            "postings can carry it."
        ),
        "check": _c(
            """
            from webledger.accounts import ChartOfAccounts
            from webledger.entries import EntryLine, JournalEntry
            from webledger.journal import Journal
            assert JournalEntry(date="2024-01-01", memo="m", lines=[]).reference == ""
            entry = JournalEntry(date="2024-01-01", memo="m", lines=[], reference="INV-1")
            assert entry.reference == "INV-1"
            journal = Journal(ChartOfAccounts().default_chart())
            posted = journal.post_simple("2024-03-01", "m", "100", "400", 500, reference="INV-2")
            assert posted.reference == "INV-2"
            """
        ),
    },
    {
        "id": "webledger-schema-02",
        "klass": "schema_or_framework_change",
        "repo": "webledger",
        "instruction": (
            "Add a memo field (default \"\") to the EntryLine dataclass in "
            "src/webledger/entries.py, and make parse_csv_lines in "
            "src/webledger/importers.py populate each line's memo with the row's "
            "entry memo."
        ),
        "check": _c(
            """
            from webledger.entries import EntryLine
            from webledger.importers import parse_csv_lines
            assert EntryLine("100", debit_cents=5, memo="wire fee").memo == "wire fee"
            assert EntryLine("100", credit_cents=5).memo == ""
            entries = parse_csv_lines("2024-03-03,fee,500,100,750\\n")
            assert entries[0].lines[0].memo == "fee"
            assert entries[0].lines[1].memo == "fee"
            """
        ),
    },
    # =====================================================================
    # biglib (33)
    # =====================================================================
    {
        "id": "biglib-lookup-01",
        "klass": "simple_lookup",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/numbers.py, distribute() puts the remainder on the left "
            "but that is undocumented and surprising. Move the remainder to the "
            "right (distribute(7, 3) == [2, 2, 3]) and update the docstring to say "
            "so."
        ),
        "check": _c(
            """
            from biglib.numbers import distribute
            assert distribute(7, 3) == [2, 2, 3]
            assert distribute(5, 2) == [2, 3]
            assert distribute(6, 3) == [2, 2, 2]
            assert "left" not in distribute.__doc__.lower()
            """
        ),
    },
    {
        "id": "biglib-lookup-02",
        "klass": "simple_lookup",
        "repo": "biglib",
        "instruction": (
            "display_name in src/biglib/strings.py capitalizes only the first "
            "character. Make it capitalize every whitespace-separated word (leaving "
            "the rest of each word untouched)."
        ),
        "check": _c(
            """
            from biglib.strings import display_name
            assert display_name("hello world") == "Hello World"
            assert display_name("  aBC   dEF ") == "ABC DEF"
            assert display_name("") == ""
            """
        ),
    },
    {
        "id": "biglib-lookup-03",
        "klass": "simple_lookup",
        "repo": "biglib",
        "instruction": (
            "column_totals in src/biglib/tables.py silently zero-fills ragged rows "
            "even though its docstring says so only as a side effect. Make it raise "
            "ValueError when rows have differing lengths instead, and update the "
            "docstring and any tests."
        ),
        "check": _c(
            """
            from biglib.tables import column_totals
            assert column_totals([[1, 2], [3, 4]]) == [4, 6]
            assert column_totals([]) == []
            try:
                column_totals([[1, 2], [3]])
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """
        ),
    },
    {
        "id": "biglib-cross-01",
        "klass": "cross_file_bug",
        "repo": "biglib",
        "instruction": (
            "asset_name in src/biglib/strings.py converts via camel_case, which "
            "glues words together before filename normalization ('quarterly "
            "summary' becomes 'quarterlysummary.txt'). Use the word-preserving "
            "kebab path from biglib.large instead, so asset_name('quarterly "
            "summary') == 'quarterly-summary.txt', and update the test that pins "
            "the old behavior."
        ),
        "check": _c(
            """
            from biglib.strings import asset_name
            assert asset_name("quarterly summary") == "quarterly-summary.txt"
            assert asset_name("Q3 Report.PDF") == "q3-report.pdf"
            tests_text = open(os.path.join(root, "tests", "test_biglib.py")).read()
            assert "quarterlysummary.txt" not in tests_text
            """
        ),
    },
    {
        "id": "biglib-cross-02",
        "klass": "cross_file_bug",
        "repo": "biglib",
        "instruction": (
            "display_name in src/biglib/strings.py truncates mid-word when the text "
            "exceeds max_length. When truncation is needed, cut back to the last "
            "word boundary that fits (dropping the trailing partial word) before "
            "returning."
        ),
        "check": _c(
            """
            from biglib.strings import display_name
            assert display_name("hello brave world", max_length=8) == "Hello"
            assert display_name("hello brave world", max_length=11) == "Hello brave"
            assert display_name("abcdefgh", max_length=5) == "Abcde"
            assert display_name("hi there", max_length=20) == "Hi there"
            """
        ),
    },
    {
        "id": "biglib-feature-01",
        "klass": "feature_addition",
        "repo": "biglib",
        "instruction": (
            "Add a stdev(values) function to src/biglib/stats.py returning the "
            "population standard deviation (square root of variance); it must raise "
            "ValueError for fewer than two values, like variance does."
        ),
        "check": _c(
            """
            from biglib.stats import stdev
            assert stdev([0, 0, 10, 10]) == 5.0
            assert abs(stdev([2, 4, 4, 4, 5, 5, 7, 9]) - 2.0) < 1e-9
            try:
                stdev([1])
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            """
        ),
    },
    {
        "id": "biglib-feature-02",
        "klass": "feature_addition",
        "repo": "biglib",
        "instruction": (
            "Add render_csv(headers, rows) to src/biglib/tables.py producing CSV text "
            "(comma-separated, one line per row) with proper double-quote escaping: "
            "cells containing commas or quotes are wrapped in quotes (internal "
            "quotes doubled)."
        ),
        "check": _c(
            """
            from biglib.tables import render_csv
            assert render_csv(["a", "b"], [["1", "x,y"]]) == 'a,b\\n1,"x,y"'
            assert render_csv(["a"], [['say "hi' ' there now']]) == 'a\\n"say ""hi there now"'
            assert render_csv(["a"], [["plain"]]) == "a\\nplain"
            """
        ),
    },
    {
        "id": "biglib-refactor-01",
        "klass": "refactor",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, legacy_wrap duplicates wrap_text's greedy "
            "algorithm. Reimplement legacy_wrap as a thin delegation to wrap_text, "
            "preserving its exact output for all inputs."
        ),
        "check": _c(
            """
            import inspect
            from biglib import large
            source = inspect.getsource(large.legacy_wrap)
            assert "wrap_text(" in source, source
            assert large.legacy_wrap("a bb ccc", 3) == "a\\nbb\\nccc"
            assert large.legacy_wrap("one two three", 7) == large.wrap_text("one two three", 7)
            """
        ),
    },
    {
        "id": "biglib-refactor-02",
        "klass": "refactor",
        "repo": "biglib",
        "instruction": (
            "Rename the SEP_RE constant in src/biglib/large.py to WORD_SPLIT_RE and "
            "update every use within the module (and anywhere else it is imported) "
            "so no SEP_RE reference remains."
        ),
        "check": _c(
            """
            from biglib import large
            assert not hasattr(large, "SEP_RE")
            assert hasattr(large, "WORD_SPLIT_RE")
            assert large.slugify("Hello World") == "hello-world"
            text = open(os.path.join(root, "src", "biglib", "large.py")).read()
            assert "SEP_RE" not in text
            """
        ),
    },
    {
        "id": "biglib-refactor-03",
        "klass": "refactor",
        "repo": "biglib",
        "instruction": (
            "pad_left, pad_right and pad_center in src/biglib/large.py repeat the "
            "same padding call three times. Extract a single private helper _pad(text, "
            "width, fill, align) and route all three public functions through it, "
            "keeping behavior identical."
        ),
        "check": _c(
            """
            import inspect
            from biglib import large
            assert "def _pad" in inspect.getsource(large)
            for name in ("pad_left", "pad_right", "pad_center"):
                assert "_pad(" in inspect.getsource(getattr(large, name)), name
            assert large.pad_left("7", 3) == "  7"
            assert large.pad_right("7", 3, ".") == "7.."
            assert large.pad_center("hi", 6, "*") == "**hi**"
            """
        ),
    },
    {
        "id": "biglib-api-01",
        "klass": "api_signature_propagation",
        "repo": "biglib",
        "instruction": (
            "Add a separator keyword parameter (default \"-\") to slugify in "
            "src/biglib/large.py controlling the joining character, and update any "
            "internal callers so they keep the dash default."
        ),
        "check": _c(
            """
            from biglib.large import slugify
            assert slugify("Hello World", separator="_") == "hello_world"
            assert slugify("Hello World") == "hello-world"
            assert slugify("Hello, World! 2024", separator="+") == "hello+world+2024"
            """
        ),
    },
    {
        "id": "biglib-api-02",
        "klass": "api_signature_propagation",
        "repo": "biglib",
        "instruction": (
            "Add a break_long_words keyword parameter (default True) to wrap_text in "
            "src/biglib/large.py: when False, words longer than the width are kept "
            "intact on their own line instead of being split. Default behavior must "
            "be byte-identical for existing callers."
        ),
        "check": _c(
            """
            from biglib.large import wrap_text
            assert wrap_text("abcdefgh x", 5, break_long_words=False) == "abcdefgh\\nx"
            assert wrap_text("abcdefgh x", 5) == "abcdefgh\\nx"
            assert wrap_text("one two three four", 7) == "one two\\nthree\\nfour"
            """
        ),
    },
    {
        "id": "biglib-api-03",
        "klass": "api_signature_propagation",
        "repo": "biglib",
        "instruction": (
            "Add an ellipsis keyword parameter (default None) to truncate in "
            "src/biglib/large.py: when given and the text exceeds the limit, the "
            "result is truncated so that the ellipsis fits within limit characters. "
            "Without it, hard truncation is unchanged."
        ),
        "check": _c(
            """
            from biglib.large import truncate
            assert truncate("abcdef", 5, ellipsis="...") == "ab..."
            assert truncate("abcdef", 5) == "abcde"
            assert truncate("abcdef", 3) == "abc"
            assert truncate("abc", 5, ellipsis="...") == "abc"
            """
        ),
    },
    {
        "id": "biglib-api-04",
        "klass": "api_signature_propagation",
        "repo": "biglib",
        "instruction": (
            "Add a keep_blank keyword parameter (default False) to parse_query_string "
            "in src/biglib/large.py: by default pairs with an empty value are "
            "dropped; with keep_blank=True they are kept with empty string values. "
            "Update callers as needed."
        ),
        "check": _c(
            """
            from biglib.large import parse_query_string
            assert parse_query_string("a=1&b=") == {"a": "1"}
            assert parse_query_string("a=1&b=", keep_blank=True) == {"a": "1", "b": ""}
            assert parse_query_string("a=1&b=2") == {"a": "1", "b": "2"}
            """
        ),
    },
    {
        "id": "biglib-api-05",
        "klass": "api_signature_propagation",
        "repo": "biglib",
        "instruction": (
            "Add an include_variance keyword parameter (default True) to summarize in "
            "src/biglib/stats.py: when False the returned dict omits the 'variance' "
            "key. Existing calls are unchanged."
        ),
        "check": _c(
            """
            from biglib.stats import summarize
            assert summarize([1, 2, 3, 4], include_variance=False) == {"mean": 2.5, "median": 2.5}
            assert summarize([1, 2, 3, 4]) == {"mean": 2.5, "median": 2.5, "variance": 1.25}
            """
        ),
    },
    {
        "id": "biglib-schema-01",
        "klass": "schema_or_framework_change",
        "repo": "biglib",
        "instruction": (
            "TextBuffer in src/biglib/large.py gains an iteration contract: add "
            "__iter__ yielding its lines and a lines property returning the current "
            "lines as a new list."
        ),
        "check": _c(
            """
            from biglib.large import TextBuffer
            buf = TextBuffer("a\\nb")
            assert list(buf) == ["a", "b"]
            assert buf.lines == ["a", "b"]
            buf.insert_line(1, "1.5")
            assert buf.lines == ["a", "1.5", "b"]
            """
        ),
    },
    {
        "id": "biglib-schema-02",
        "klass": "schema_or_framework_change",
        "repo": "biglib",
        "instruction": (
            "Give TextBuffer in src/biglib/large.py a readonly constructor flag "
            "(default False): when set, insert_line and replace_line raise "
            "RuntimeError('buffer is readonly') instead of mutating."
        ),
        "check": _c(
            """
            from biglib.large import TextBuffer
            frozen = TextBuffer("x", readonly=True)
            try:
                frozen.insert_line(0, "y")
                raise AssertionError("expected RuntimeError")
            except RuntimeError as exc:
                assert "readonly" in str(exc)
            normal = TextBuffer("x")
            normal.insert_line(0, "y")
            assert normal.text() == "y\\nx"
            """
        ),
    },
    {
        "id": "biglib-schema-03",
        "klass": "schema_or_framework_change",
        "repo": "biglib",
        "instruction": (
            "Change summarize in src/biglib/stats.py to return a new Summary "
            "dataclass (fields mean, median, variance) instead of a dict, and update "
            "its consumers plus tests so the fixture suite passes with the new "
            "return type."
        ),
        "check": _c(
            """
            from biglib.stats import Summary, summarize
            summary = summarize([1, 2, 3, 4])
            assert isinstance(summary, Summary)
            assert summary.mean == 2.5 and summary.median == 2.5 and summary.variance == 1.25
            """
        ),
    },
    {
        "id": "biglib-large-01",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In the large module src/biglib/large.py, format_ingest_label only "
            "capitalizes the first character of the whole label. Find it and make it "
            "capitalize the first letter of every whitespace-separated word instead."
        ),
        "check": _c(
            """
            from biglib.large import format_ingest_label
            assert format_ingest_label("hello world") == "Hello World"
            assert format_ingest_label("  ingest PREVIEW text ") == "Ingest Preview Text"
            assert format_ingest_label("") == ""
            """
        ),
    },
    {
        "id": "biglib-large-02",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, locate count_index_tokens and make it ignore "
            "the common stopwords 'the', 'a' and 'of' (case-insensitive) when "
            "counting whitespace-separated tokens."
        ),
        "check": _c(
            """
            from biglib.large import count_index_tokens
            assert count_index_tokens("the cat and a dog") == 3
            assert count_index_tokens("The of a") == 0
            assert count_index_tokens("state-of-the-art") == 1
            assert count_index_tokens("") == 0
            """
        ),
    },
    {
        "id": "biglib-large-03",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, is_notify_empty treats separator-only strings "
            "like ' - _ ' as non-empty because it only strips whitespace. Make it "
            "return True for text containing only whitespace, dashes and "
            "underscores."
        ),
        "check": _c(
            """
            from biglib.large import is_notify_empty
            assert is_notify_empty(" - _ ") is True
            assert is_notify_empty("-__-") is True
            assert is_notify_empty("") is True
            assert is_notify_empty("ok") is False
            assert is_notify_empty("a-b") is False
            """
        ),
    },
    {
        "id": "biglib-large-04",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, strip_archive_prefix matches its prefix "
            "case-sensitively. Make the prefix comparison case-insensitive so "
            "'Archive: log' with prefix 'archive:' is stripped."
        ),
        "check": _c(
            """
            from biglib.large import strip_archive_prefix
            assert strip_archive_prefix("Archive: log", "archive:") == " log"
            assert strip_archive_prefix("ARCHIVE: log", "archive:") == " log"
            assert strip_archive_prefix("archive:log", "archive:") == "log"
            assert strip_archive_prefix("keep: log", "archive:") == "keep: log"
            """
        ),
    },
    {
        "id": "biglib-large-05",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, append_tagging_suffix produces doubled "
            "separators when the text ends with a dash and the suffix starts with "
            "one. Collapse the boundary so only a single separator remains."
        ),
        "check": _c(
            """
            from biglib.large import append_tagging_suffix
            assert append_tagging_suffix("done-", "-x") == "done-x"
            assert append_tagging_suffix("done", "-x") == "done-x"
            assert append_tagging_suffix("done-", "x") == "done-x"
            assert append_tagging_suffix("done-x", "-x") == "done-x"
            """
        ),
    },
    {
        "id": "biglib-large-06",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, the IngestPreview class splits on underscores "
            "via the shared regex, destroying snake_case words. Change only "
            "IngestPreview.split so it splits on whitespace and hyphens but preserves "
            "underscores."
        ),
        "check": _c(
            """
            from biglib.large import IngestPreview, snake_case
            preview = IngestPreview()
            assert preview.build("a_b c") == "a_b-c"
            assert preview.split("hello-world two") == ["hello", "world", "two"]
            assert preview.count("a_b") == 1
            assert snake_case("Hello World") == "hello_world"
            """
        ),
    },
    {
        "id": "biglib-large-07",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, normalize_digest leaves a leading '#' marker in "
            "place. Strip one leading '#' (plus following whitespace) as part of its "
            "normalization."
        ),
        "check": _c(
            """
            from biglib.large import normalize_digest
            assert normalize_digest("# Title Here") == "title here"
            assert normalize_digest("#Title") == "title"
            assert normalize_digest("plain text") == "plain text"
            """
        ),
    },
    {
        "id": "biglib-large-08",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "Somewhere in the big module src/biglib/large.py there is a dead helper "
            "named unused_unescape that nothing calls. Delete it and remove the test "
            "assertion that pins it in tests/test_biglib.py so the suite still "
            "passes."
        ),
        "check": _c(
            """
            from biglib import large
            assert not hasattr(large, "unused_unescape")
            assert hasattr(large, "legacy_wrap") and hasattr(large, "old_slugify")
            tests_text = open(os.path.join(root, "tests", "test_biglib.py")).read()
            assert "unused_unescape" not in tests_text
            """
        ),
    },
    {
        "id": "biglib-large-09",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, truncate silently accepts negative limits "
            "(Python slicing makes truncate('abc', -1) return 'ab'). Raise "
            "ValueError for negative limits instead."
        ),
        "check": _c(
            """
            from biglib import large
            try:
                large.truncate("abc", -1)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
            assert large.truncate("abc", 2) == "ab"
            assert large.truncate("abc", 10) == "abc"
            """
        ),
    },
    {
        "id": "biglib-large-10",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, parse_query_string keeps '+' literal in values. "
            "Decode '+' as a space in values (query-string convention) while leaving "
            "keys untouched."
        ),
        "check": _c(
            """
            from biglib.large import parse_query_string
            assert parse_query_string("q=a+b")["q"] == "a b"
            assert parse_query_string("x+y=1")["x+y"] == "1"
            assert parse_query_string("a=1&b=2") == {"a": "1", "b": "2"}
            """
        ),
    },
    {
        "id": "biglib-large-11",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, TextBuffer.replace_line lets list's generic "
            "IndexError escape. Give out-of-range replacements a precise error "
            "message containing 'index <n> out of range' (e.g. 'index 99 out of "
            "range')."
        ),
        "check": _c(
            """
            from biglib.large import TextBuffer
            buf = TextBuffer("one\\ntwo")
            try:
                buf.replace_line(99, "x")
                raise AssertionError("expected IndexError")
            except IndexError as exc:
                assert "index 99 out of range" in str(exc), exc
            buf.replace_line(0, "ONE")
            assert buf.text() == "ONE\\ntwo"
            """
        ),
    },
    {
        "id": "biglib-large-12",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, old_slugify mangles accented characters ('Café "
            "Test' becomes 'caf-test'). Make it delegate to slugify so unicode text "
            "normalizes identically."
        ),
        "check": _c(
            """
            from biglib import large
            assert large.old_slugify("Café Test") == "cafe-test"
            assert large.old_slugify("Hello World") == "hello-world"
            assert large.old_slugify("a__b") == "a-b"
            """
        ),
    },
    {
        "id": "biglib-large-13",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, strip_markup_prefix removes only one leading "
            "occurrence of the prefix. Make it strip ALL leading occurrences so "
            "strip_markup_prefix('---x', '-') == 'x'."
        ),
        "check": _c(
            """
            from biglib.large import strip_markup_prefix
            assert strip_markup_prefix("---x", "-") == "x"
            assert strip_markup_prefix("-x", "-") == "x"
            assert strip_markup_prefix("x", "-") == "x"
            assert strip_markup_prefix("a-b", "-") == "a-b"
            """
        ),
    },
    {
        "id": "biglib-large-14",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, the RenderOutput.join method keeps parts that "
            "consist only of the separator, producing runs like 'a---b'. Make join "
            "skip parts equal to the separator itself."
        ),
        "check": _c(
            """
            from biglib.large import RenderOutput
            out = RenderOutput()
            assert out.join(["a", "-", "b"]) == "a-b"
            assert out.join(["a", "", "b"]) == "a-b"
            assert out.join(["a", "b"]) == "a-b"
            assert out.build("hello world") == "hello-world"
            """
        ),
    },
    {
        "id": "biglib-large-15",
        "klass": "large_file_navigation",
        "repo": "biglib",
        "instruction": (
            "In src/biglib/large.py, truncate_with_ellipsis cuts mid-word. When the "
            "truncation point lands inside a word, cut back to the end of the last "
            "complete word before appending the ellipsis (a single long first word "
            "is still hard-truncated)."
        ),
        "check": _c(
            """
            from biglib.large import truncate_with_ellipsis
            assert truncate_with_ellipsis("hello world foo", 9) == "hello..."
            assert truncate_with_ellipsis("abcdef", 5) == "ab..."
            assert truncate_with_ellipsis("abc", 5) == "abc"
            assert truncate_with_ellipsis("one two", 3) == "..."
            """
        ),
    },
]

SAMPLE_SOLUTIONS = [
    "shopcart-lookup-01",
    "shopcart-cross-03",
    "shopcart-api-02",
    "pipeline-cross-02",
    "pipeline-feature-03",
    "notesrv-lookup-01",
    "notesrv-schema-01",
    "webledger-lookup-01",
    "webledger-refactor-02",
    "webledger-api-04",
    "biglib-large-01",
    "biglib-api-01",
]

STRATUM_ORDER = [
    "simple_lookup",
    "cross_file_bug",
    "feature_addition",
    "refactor",
    "api_signature_propagation",
    "schema_or_framework_change",
    "large_file_navigation",
]


def task_by_id(task_id: str) -> dict:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise SystemExit(f"unknown task {task_id}")


def fixture_root(repo: str) -> Path:
    return FIXTURES / repo


def copy_fixture(repo: str, dest: Path) -> Path:
    shutil.copytree(
        fixture_root(repo),
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return dest


def run_check(task_id: str, copy_root: Path) -> tuple[bool, str]:
    """Exec one check in this process with root bound; returns (ok, detail)."""
    task = task_by_id(task_id)
    scope: dict = {"root": str(copy_root)}
    try:
        code = compile(task["check"], f"<check:{task_id}>", "exec")
        exec(code, scope)  # noqa: S102 - benchmark checkers are first-party
        return bool(scope.get("ok", False)), scope.get("detail", "")
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported, not raised
        return False, f"{type(exc).__name__}: {exc}"


def run_check_subprocess(task_id: str, copy_root: Path) -> tuple[bool, str]:
    """Run a check in a fresh interpreter so fixture modules never leak between checks."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-check", task_id, str(copy_root)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(copy_root),
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-500:]


def run_fixture_suite(copy_root: Path) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONPATH=str(copy_root / "src"))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(copy_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, proc.stdout[-500:]


def cmd_validate() -> int:
    """Every check must FAIL against a pristine copy of its fixture."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for index, task in enumerate(TASKS, 1):
            copy_root = Path(td) / task["repo"] / task["id"]
            copy_fixture(task["repo"], copy_root)
            ok, detail = run_check_subprocess(task["id"], copy_root)
            status = "PASS(bad)" if ok else "fail(good)"
            print(f"[{index:>3}/{len(TASKS)}] {task['id']:<28} {status}")
            if ok:
                failures.append(f"{task['id']}: check passed on pristine fixture")
    if failures:
        print(f"\nVALIDATION FAILED: {len(failures)} check(s) passed on pristine:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"\nAll {len(TASKS)} checks fail on pristine fixtures. OK")
    return 0


def cmd_solve() -> int:
    """Apply reference diffs for the sampled tasks; checks must then pass."""
    missing = [t for t in SAMPLE_SOLUTIONS if not (SOLUTIONS / f"{t}.diff").exists()]
    if missing:
        print(f"missing reference diffs: {missing}")
        return 1
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for task_id in SAMPLE_SOLUTIONS:
            task = task_by_id(task_id)
            copy_root = Path(td) / task["repo"] / task_id
            copy_fixture(task["repo"], copy_root)
            proc = subprocess.run(
                ["git", "apply", "-p1", str(SOLUTIONS / f"{task_id}.diff")],
                cwd=str(copy_root),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                failures.append(f"{task_id}: diff apply failed: {proc.stderr.strip()[-300:]}")
                continue
            ok, detail = run_check_subprocess(task_id, copy_root)
            if not ok:
                failures.append(f"{task_id}: check still fails after solution: {detail}")
                continue
            suite_ok, suite_detail = run_fixture_suite(copy_root)
            if not suite_ok:
                failures.append(f"{task_id}: fixture suite fails after solution: {suite_detail}")
                continue
            print(f"{task_id:<28} solution applies, check passes, suite passes")
    if failures:
        print("\nSOLVE FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"\nAll {len(SAMPLE_SOLUTIONS)} sampled reference solutions verified. OK")
    return 0


def cmd_emit() -> int:
    counts: dict[str, int] = {}
    for task in TASKS:
        counts[task["klass"]] = counts.get(task["klass"], 0) + 1
    payload = [
        {
            "id": task["id"],
            "klass": task["klass"],
            "repo": task["repo"],
            "instruction": task["instruction"],
            "check": task["check"],
        }
        for task in TASKS
    ]
    EMIT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EMIT_PATH} with {len(payload)} tasks")
    for klass in STRATUM_ORDER:
        print(f"  {klass:<28} {counts.get(klass, 0)}")
    return 0


def cmd_list() -> int:
    for task in TASKS:
        print(f"{task['id']:<28} {task['klass']:<28} {task['repo']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="gen_tasks")
    parser.add_argument("--validate", action="store_true", help="all checks must fail on pristine fixtures")
    parser.add_argument("--solve", action="store_true", help="apply sampled reference diffs and verify checks pass")
    parser.add_argument("--emit", action="store_true", help="write tasks_generated.json")
    parser.add_argument("--list", action="store_true", help="list all tasks")
    parser.add_argument("--run-check", nargs=2, metavar=("TASK_ID", "ROOT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_check:
        task_id, root = args.run_check
        ok, detail = run_check(task_id, Path(root))
        if detail:
            print(detail)
        print(f"check {'passed' if ok else 'failed'}: {task_id}")
        return 0 if ok else 1
    if args.list:
        return cmd_list()

    ran = False
    code = 0
    if args.validate:
        ran = True
        code = cmd_validate()
    if args.solve:
        ran = True
        code = cmd_solve() or code
    if args.emit:
        ran = True
        code = cmd_emit() or code
    if not ran:
        parser.print_help()
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())

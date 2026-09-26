from shopcart.discounts import (
    DISCOUNT_CODES,
    apply_code,
    legacy_percent_off,
    lookup_code,
    validate_code,
)


def test_lookup_is_case_insensitive():
    assert lookup_code("save10") is DISCOUNT_CODES["SAVE10"]
    assert lookup_code("nope") is None


def test_apply_percent_code():
    new_subtotal, message = apply_code(10000, "SAVE10")
    assert new_subtotal == 9000
    assert message == "applied SAVE10"


def test_apply_flat_code():
    new_subtotal, _ = apply_code(2000, "FIVEOFF")
    assert new_subtotal == 1500


def test_min_subtotal_gate():
    ok, reason = validate_code(DISCOUNT_CODES["BIGDEAL"], 5000)
    assert not ok
    assert "minimum" in reason
    new_subtotal, message = apply_code(5000, "BIGDEAL")
    assert new_subtotal == 5000
    assert "not applicable" in message


def test_unknown_code_is_noop():
    new_subtotal, message = apply_code(1000, "GHOST")
    assert new_subtotal == 1000
    assert message == "unknown code"


def test_legacy_helper_still_matches_int_math():
    # Kept for the historical importer; percent is clamped at 100.
    assert legacy_percent_off(1000, 10) == 900
    assert legacy_percent_off(1000, 150) == 0

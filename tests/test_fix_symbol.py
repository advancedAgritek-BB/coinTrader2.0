import pytest

from crypto_bot.main import _fix_symbol as fix_symbol


def test_fix_symbol_converts_xbt():
    assert fix_symbol("XBT/USDT") == "BTC/USDT"


@pytest.mark.parametrize("value", [None, 42, 1.0, [], {"a": 1}])
def test_fix_symbol_passthrough_non_string(value):
    assert fix_symbol(value) == value

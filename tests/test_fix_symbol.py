import pytest

from crypto_bot.main import _fix_symbol as fix_symbol


@pytest.mark.parametrize(
    "input_sym",
    ["XBT/USDT", "XBTUSDT", "BTC/USDT"],
)
def test_fix_symbol_converts_xbt(input_sym):
    assert fix_symbol(input_sym) == "BTCUSDT"


@pytest.mark.parametrize("value", [None, 42, 1.0, [], {"a": 1}])
def test_fix_symbol_passthrough_non_string(value):
    assert fix_symbol(value) == value

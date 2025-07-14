import asyncio
import sys
import types

import pytest

from crypto_bot.solana import pyth_utils

class DummyEnum:
    PRICE = "PRICE"


class DummyPriceAcc:
    def __init__(self, price):
        self.aggregate_price = price


class DummyProduct:
    def __init__(self, symbol, price):
        self.symbol = symbol
        self._prices = {DummyEnum.PRICE: DummyPriceAcc(price)}

    async def get_prices(self):
        return self._prices


class DummyClient:
    def __init__(self, products, fail=False):
        self._products = products
        self.refresh_called = False
        self.fail = fail

    async def refresh_all_prices(self):
        self.refresh_called = True
        if self.fail:
            raise RuntimeError("boom")

    @property
    def products(self):
        return self._products


def test_get_pyth_price_ok():
    sys.modules["pythclient.pythaccounts"] = types.SimpleNamespace(PythPriceType=DummyEnum)
    client = DummyClient([DummyProduct("SOL/USD", 42.0)])
    price = asyncio.run(pyth_utils.get_pyth_price(client, "SOL/USD"))
    assert price == 42.0
    assert client.refresh_called


def test_get_pyth_price_error():
    sys.modules["pythclient.pythaccounts"] = types.SimpleNamespace(PythPriceType=DummyEnum)
    client = DummyClient([DummyProduct("SOL/USD", 42.0)], fail=True)
    with pytest.raises(RuntimeError):
        asyncio.run(pyth_utils.get_pyth_price(client, "SOL/USD"))

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
import importlib
import sys


class DummyPrice:
    def __init__(self, price, conf, status):
        self.aggregate_price = price
        self.aggregate_price_confidence_interval = conf
        self.aggregate_price_status = status


class DummyProduct:
    def __init__(self, symbol, price_obj):
        self.symbol = symbol
        self.prices = {"PRICE": price_obj}


class DummyClient:
    def __init__(self, *args, **kwargs):
        DummyClient.last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def refresh_all_prices(self):
        pass

    async def get_products(self):
        return [DummyProduct("BTC/USD", DummyPrice(10.0, 0.2, DummyStatus.TRADING))]


class DummyStatus:
    TRADING = "TRADING"


def test_get_pyth_price(monkeypatch):
    # Provide minimal aiohttp stub before importing module
    monkeypatch.setitem(sys.modules, "aiohttp", type("M", (), {"ClientError": Exception}))

    # Patch modules imported inside the utility
    dummy_mod = type("M", (), {"PythClient": DummyClient})
    monkeypatch.setitem(sys.modules, "pythclient.pythclient", dummy_mod)
    monkeypatch.setitem(
        sys.modules,
        "pythclient.pythaccounts",
        type("S", (), {"PythPriceStatus": DummyStatus}),
    )
    monkeypatch.setitem(
        sys.modules,
        "pythclient.utils",
        type("U", (), {"get_key": lambda n, t: "k"}),
    )

    pyth_utils = importlib.import_module("crypto_bot.utils.pyth_utils")

    cfg = {
        "http_endpoint": "http://rpc",
        "ws_endpoint": "ws://rpc",
        "mapping_key": "m",
        "program_key": "p",
    }

    price, conf_pct, trading = asyncio.run(pyth_utils.get_pyth_price("BTC/USD", cfg))

    assert price == 10.0
    assert conf_pct == 0.02
    assert trading is True
    assert DummyClient.last_kwargs["solana_endpoint"] == "http://rpc"
    assert DummyClient.last_kwargs["solana_ws_endpoint"] == "ws://rpc"

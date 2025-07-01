import asyncio

import pandas as pd
import pytest

from crypto_bot.portfolio_rotator import PortfolioRotator


class DummyExchange:
    def __init__(self, data):
        self.data = data

    def fetch_ohlcv(self, symbol, timeframe="1d", limit=30):
        closes = self.data[symbol]
        return [[0, c, c, c, c, 1] for c in closes[-limit:]]


def test_score_assets_returns_scores():
    data = {
        "BTC": [1, 2, 3, 4, 5],
        "ETH": [1, 1.5, 2, 2.5, 3],
    }
    ex = DummyExchange(data)
    rotator = PortfolioRotator()
    scores = rotator.score_assets(ex, ["BTC", "ETH"], 5, "momentum")
    assert set(scores.keys()) == {"BTC", "ETH"}
    assert scores["BTC"] > scores["ETH"]


def test_rotate_calls_converter(monkeypatch):
    rotator = PortfolioRotator()
    rotator.config.update({"scoring_method": "sharpe", "top_assets": 1, "rebalance_threshold": 0.1})

    called = {}

    async def fake_convert(wallet, from_t, to_t, amt, dry_run=True):
        called["args"] = (wallet, from_t, to_t, amt, dry_run)
        return {}

    monkeypatch.setattr("crypto_bot.portfolio_rotator.auto_convert_funds", fake_convert)
    monkeypatch.setattr(rotator, "score_assets", lambda *a, **k: {"BTC": 0.5, "ETH": 0.1})

    holdings = {"ETH": 10}
    new_holdings = rotator.rotate(object(), "wallet", holdings)

    assert called["args"][1] == "ETH"
    assert called["args"][2] == "BTC"
    assert new_holdings["BTC"] == 10

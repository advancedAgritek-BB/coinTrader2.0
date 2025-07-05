import asyncio
import json
import logging

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

    async def fake_convert(wallet, from_t, to_t, amt, dry_run=True, **kwargs):
        called["args"] = (wallet, from_t, to_t, amt, dry_run)
        called["kwargs"] = kwargs
        return {}

    monkeypatch.setattr("crypto_bot.portfolio_rotator.auto_convert_funds", fake_convert)
    monkeypatch.setattr(rotator, "score_assets", lambda *a, **k: {"BTC": 0.5, "ETH": 0.1})

    holdings = {"ETH": 10}
    new_holdings = asyncio.run(
        rotator.rotate(object(), "wallet", holdings)
    )

    assert called["args"][1] == "ETH"
    assert called["args"][2] == "BTC"
    assert "notifier" in called["kwargs"]
    assert called["kwargs"]["notifier"] is None
    assert new_holdings["BTC"] == 10


def test_rotate_logs_scores(tmp_path, monkeypatch):
    rotator = PortfolioRotator()
    score_file = tmp_path / "scores.json"
    monkeypatch.setattr("crypto_bot.portfolio_rotator.SCORE_FILE", score_file)
    monkeypatch.setattr(
        rotator,
        "score_assets",
        lambda *a, **k: {"BTC": 0.5, "ETH": 0.1},
    )
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator.auto_convert_funds", lambda *a, **k: {}
    )

    holdings = {"ETH": 10}
    asyncio.run(rotator.rotate(object(), "wallet", holdings))

    data = json.loads(score_file.read_text())
    assert data["BTC"] == 0.5


def test_score_assets_handles_invalid_data(caplog):
    class BadExchange:
        def fetch_ohlcv(self, symbol, timeframe="1d", limit=30):
            return None

    rotator = PortfolioRotator()
    caplog.set_level(logging.ERROR)

    scores = rotator.score_assets(BadExchange(), ["BTC"], 5, "momentum")

    assert scores == {}
    assert any("Invalid OHLCV" in r.getMessage() for r in caplog.records)

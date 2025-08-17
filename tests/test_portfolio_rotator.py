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
        base = symbol.split("/")[0]
        closes = self.data[base]
        return [[0, c, c, c, c, 1] for c in closes[-limit:]]


def test_score_assets_returns_scores():
    data = {
        "BTC": [1, 2, 3, 4, 5],
        "ETH": [1, 1.5, 2, 2.5, 3],
    }
    ex = DummyExchange(data)
    rotator = PortfolioRotator()
    scores = asyncio.run(
        rotator.score_assets(ex, ["BTC/USD", "ETH/USD"], 5, "momentum")
    )
    assert set(scores.keys()) == {"BTC/USD", "ETH/USD"}
    assert scores["BTC/USD"] > scores["ETH/USD"]


def test_rotate_calls_converter(monkeypatch):
    rotator = PortfolioRotator()
    rotator.config.update({"scoring_method": "sharpe", "top_assets": 1, "rebalance_threshold": 0.1})

    called = {}

    async def fake_convert(wallet, from_t, to_t, amt, dry_run=True, **kwargs):
        called["args"] = (wallet, from_t, to_t, amt, dry_run)
        called["kwargs"] = kwargs
        return {}

    monkeypatch.setattr("crypto_bot.portfolio_rotator.auto_convert_funds", fake_convert)

    async def fake_scores(*a, **k):
        return {"BTC": 0.5, "ETH": 0.1}

    monkeypatch.setattr(rotator, "score_assets", fake_scores)
    async def fake_score_assets(*_a, **_k):
        return {"BTC": 0.5, "ETH": 0.1}

    monkeypatch.setattr(rotator, "score_assets", fake_score_assets)

    exchange = type("Ex", (), {"symbols": ["ETH/USD", "BTC/USD"]})()
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator._pairs_from_balance",
        lambda ex, bal, quotes=None: ["ETH/USD", "BTC/USD"],
    )

    holdings = {"ETH": 10}
    new_holdings = asyncio.run(rotator.rotate(exchange, "wallet", holdings))

    assert called["args"][1] == "ETH"
    assert called["args"][2] == "BTC"
    assert "notifier" in called["kwargs"]
    assert called["kwargs"]["notifier"] is None
    assert new_holdings["BTC"] == 10


def test_rotate_logs_scores(tmp_path, monkeypatch):
    rotator = PortfolioRotator()
    score_file = tmp_path / "scores.json"
    monkeypatch.setattr("crypto_bot.portfolio_rotator.SCORE_FILE", score_file)
    async def fake_scores(*a, **k):
        return {"BTC": 0.5, "ETH": 0.1}

    monkeypatch.setattr(rotator, "score_assets", fake_scores)
    async def fake_score_assets(*_a, **_k):
        return {"BTC": 0.5, "ETH": 0.1}

    monkeypatch.setattr(rotator, "score_assets", fake_score_assets)
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator.auto_convert_funds", lambda *a, **k: {}
    )

    exchange = type("Ex", (), {"symbols": ["ETH/USD", "BTC/USD"]})()
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator._pairs_from_balance",
        lambda ex, bal, quotes=None: ["ETH/USD", "BTC/USD"],
    )

    holdings = {"ETH": 10}
    asyncio.run(rotator.rotate(exchange, "wallet", holdings))

    data = json.loads(score_file.read_text())
    assert data["BTC"] == 0.5


class BadDataExchange(DummyExchange):
    def fetch_ohlcv(self, symbol, timeframe="1d", limit=30):
        if symbol.split("/")[0] == "BAD":
            raise ValueError("bad data")
        return super().fetch_ohlcv(symbol, timeframe, limit)


def test_score_assets_skips_invalid_ohlcv(caplog):
    data = {"BTC": [1, 2, 3], "BAD": [1, 2, 3]}
    ex = BadDataExchange(data)
    rotator = PortfolioRotator()

    with caplog.at_level("ERROR"):
        scores = asyncio.run(
            rotator.score_assets(ex, ["BTC/USD", "BAD/USD"], 3, "momentum")
        )

    assert set(scores.keys()) == {"BTC/USD"}
    assert any("OHLCV fetch failed for BAD/USD" in r.message for r in caplog.records)


def test_rotate_ignores_tokens_without_pair(monkeypatch):
    ex = BadDataExchange({"GOOD": [1, 2, 3], "BAD": [1, 2, 3]})
    rotator = PortfolioRotator()
    rotator.config.update({"scoring_method": "momentum", "lookback_days": 3})

    called = {}

    async def fake_convert(*args, **kwargs):
        called["called"] = True
        return {}

    monkeypatch.setattr("crypto_bot.portfolio_rotator.auto_convert_funds", fake_convert)
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator._pairs_from_balance",
        lambda ex, bal, quotes=None: [],
    )
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator._fallback_watchlist", lambda ex, quotes=None, limit=50: [],
    )

    holdings = {"BAD": 5}
    result = asyncio.run(rotator.rotate(ex, "wallet", holdings))

    assert result == {}
    assert not called
def test_rotate_translates_assets_to_pairs(monkeypatch):
    rotator = PortfolioRotator()
    captured = {}

    async def fake_score_assets(ex, symbols, lookback, method):
        captured["symbols"] = list(symbols)
        return {}

    monkeypatch.setattr(rotator, "score_assets", fake_score_assets)
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator.auto_convert_funds", lambda *a, **k: {}
    )

    exchange = type("Ex", (), {"symbols": ["BTC/USD"]})()
    monkeypatch.setattr(
        "crypto_bot.portfolio_rotator._pairs_from_balance",
        lambda ex, bal, quotes=None: ["BTC/USD"],
    )

    asyncio.run(rotator.rotate(exchange, "wallet", {"BTC": 1}))

    assert captured.get("symbols") == ["BTC/USD"]
def test_score_assets_handles_invalid_data(caplog):
    class BadExchange:
        def fetch_ohlcv(self, symbol, timeframe="1d", limit=30):
            return None

    rotator = PortfolioRotator()
    caplog.set_level(logging.ERROR)

    scores = asyncio.run(
        rotator.score_assets(BadExchange(), ["BTC/USD"], 5, "momentum")
    )

    assert scores == {}
    assert any("Invalid OHLCV" in r.getMessage() for r in caplog.records)

import asyncio
import logging
import sys
import types

import pytest

# Stub out commit_lock to avoid import errors in tests
commit_lock_stub = types.ModuleType("commit_lock")
commit_lock_stub.check_and_update = lambda *a, **k: None
sys.modules.setdefault("crypto_bot.utils.commit_lock", commit_lock_stub)

# Stub out ml_utils to bypass __future__ placement issues
# Stub out ml_utils to avoid syntax errors during import
ml_utils_stub = types.ModuleType("ml_utils")
ml_utils_stub.is_ml_available = lambda: False
ml_utils_stub.ML_AVAILABLE = False
sys.modules.setdefault("crypto_bot.utils.ml_utils", ml_utils_stub)

import crypto_bot.sentiment_filter as sf


@pytest.mark.asyncio
async def test_too_bearish(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 30)
    assert await sf.too_bearish(20, 40, symbol="BTC") is True
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "90")

    async def fake_get_sentiment(symbol):
        return 30

    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment", fake_get_sentiment
    )


def test_too_bearish_sync(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    async def fake_get_sentiment(symbol):
        return 30
    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment", fake_get_sentiment
    )
    assert asyncio.run(sf.too_bearish(20, 40, symbol="BTC")) is True


@pytest.mark.asyncio
async def test_boost_factor(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 80)
    assert await sf.boost_factor(70, 60, symbol="BTC") > 1.0
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "10")

    async def fake_get_sentiment(symbol):
        return 80

    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment", fake_get_sentiment
    )


def test_boost_factor_sync(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    async def fake_get_sentiment(symbol):
        return 80
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", fake_get_sentiment)
    assert asyncio.run(sf.boost_factor(70, 60, symbol="BTC")) > 1.0


def test_fetch_twitter_sentiment_no_api_key(monkeypatch, caplog):
    sf._CACHE.clear()
    monkeypatch.delenv("LUNARCRUSH_API_KEY", raising=False)
    with caplog.at_level(logging.ERROR):
        score = sf.fetch_twitter_sentiment(symbol="ETH")
    assert score == 50
    assert any("LUNARCRUSH_API_KEY" in rec.message for rec in caplog.records)
    calls = {"n": 0}

    def fake_get(url, timeout=5):
        calls["n"] += 1

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"value": "42"}]}

        return Resp()

    monkeypatch.delenv("MOCK_FNG_VALUE", raising=False)
    monkeypatch.setattr(sf.requests, "get", fake_get)
    assert sf.fetch_fng_index() == 42
    assert sf.fetch_fng_index() == 42
    assert calls["n"] == 1


def test_fetch_twitter_sentiment_cached(monkeypatch):
    sf._CACHE.clear()
    calls = {"n": 0}

    async def fake_get_sentiment(symbol):
        calls["n"] += 1
        return 88

    monkeypatch.delenv("MOCK_TWITTER_SENTIMENT", raising=False)
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "1")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", fake_get_sentiment)
    assert sf.fetch_twitter_sentiment("eth") == 88
    assert sf.fetch_twitter_sentiment("eth") == 88
    assert calls["n"] == 1


def test_fetch_twitter_sentiment_sync_lunar(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 77)
    score = sf.fetch_twitter_sentiment(symbol="BTC")
    assert score == 77


@pytest.mark.asyncio
async def test_fetch_twitter_sentiment_async_lunar(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 66)
    score = await sf.fetch_twitter_sentiment_async(symbol="DOGE")
    assert score == 66


def test_fetch_lunarcrush_sentiment_cached(monkeypatch):
    sf._CACHE.clear()
    calls = {"n": 0}

    async def fake_get_sentiment(symbol):
        calls["n"] += 1
        return 70

    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", fake_get_sentiment)
    assert asyncio.run(sf.fetch_lunarcrush_sentiment_async("BTC")) == 70
    assert asyncio.run(sf.fetch_lunarcrush_sentiment_async("BTC")) == 70
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_too_bearish_logs(monkeypatch, caplog):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 30)
    with caplog.at_level(logging.INFO):
        await sf.too_bearish(10, 10, symbol="BTC")
    assert any("FNG 50, sentiment 30" in rec.message for rec in caplog.records)


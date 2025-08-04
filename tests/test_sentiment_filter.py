import os

import crypto_bot.sentiment_filter as sf
import pytest

from crypto_bot.sentiment_filter import too_bearish, boost_factor

import pytest

from crypto_bot.sentiment_filter import boost_factor, too_bearish


def test_too_bearish(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "90")
    monkeypatch.setattr("crypto_bot.sentiment_filter.lunar_client.get_sentiment", lambda s: 30)
    assert sf.too_bearish(20, 40, symbol="BTC") is True


def test_boost_factor(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "10")
    monkeypatch.setattr("crypto_bot.sentiment_filter.lunar_client.get_sentiment", lambda s: 80)
    assert sf.boost_factor(70, 60, symbol="BTC") > 1.0


def test_fetch_fng_index_cached(monkeypatch):
    sf._CACHE.clear()
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

    def fake_get(url, timeout=5):
        calls["n"] += 1

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"score": 88}

        return Resp()

    monkeypatch.delenv("MOCK_TWITTER_SENTIMENT", raising=False)
    monkeypatch.delenv("LUNARCRUSH_API_KEY", raising=False)
    monkeypatch.setattr(sf.requests, "get", fake_get)
    assert sf.fetch_twitter_sentiment("eth") == 88
    assert sf.fetch_twitter_sentiment("eth") == 88
    assert calls["n"] == 1


def test_fetch_lunarcrush_sentiment_cached(monkeypatch):
    sf._CACHE.clear()
    calls = {"n": 0}

    def fake_get_sentiment(symbol):
        calls["n"] += 1
        return 70

    monkeypatch.setattr(sf.lunar_client, "get_sentiment", fake_get_sentiment)
    assert sf.fetch_lunarcrush_sentiment("BTC") == 70
    assert sf.fetch_lunarcrush_sentiment("BTC") == 70
    assert calls["n"] == 1
@pytest.mark.asyncio
async def test_too_bearish(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "90")

    async def fake_get_sentiment(symbol):
        return 30

    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        fake_get_sentiment,
    )
    assert await too_bearish(20, 40, symbol="BTC") is True


@pytest.mark.asyncio
async def test_boost_factor(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "10")

    async def fake_get_sentiment(symbol):
        return 80

    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        fake_get_sentiment,
    )
    assert await boost_factor(70, 60, symbol="BTC") > 1.0


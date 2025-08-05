import asyncio
import logging

import pytest

import crypto_bot.sentiment_filter as sf


@pytest.mark.asyncio
async def test_too_bearish(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 30)
    assert await sf.too_bearish(20, 40, symbol="BTC") is True


@pytest.mark.asyncio
async def test_boost_factor(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 80)
    assert await sf.boost_factor(70, 60, symbol="BTC") > 1.0


def test_fetch_twitter_sentiment_no_api_key(monkeypatch, caplog):
    sf._CACHE.clear()
    monkeypatch.delenv("LUNARCRUSH_API_KEY", raising=False)
    with caplog.at_level(logging.ERROR):
        score = asyncio.run(sf.fetch_twitter_sentiment(symbol="ETH"))
    assert score == 50
    assert any("LUNARCRUSH_API_KEY" in rec.message for rec in caplog.records)


def test_fetch_twitter_sentiment_sync_lunar(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 77)
    score = asyncio.run(sf.fetch_twitter_sentiment(symbol="BTC"))
    assert score == 77


@pytest.mark.asyncio
async def test_fetch_twitter_sentiment_async_lunar(monkeypatch):
    sf._CACHE.clear()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "key")
    monkeypatch.setattr(sf.lunar_client, "get_sentiment", lambda s: 66)
    score = await sf.fetch_twitter_sentiment(symbol="DOGE")
    assert score == 66

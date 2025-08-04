import os

import pytest

from crypto_bot.sentiment_filter import boost_factor, too_bearish


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


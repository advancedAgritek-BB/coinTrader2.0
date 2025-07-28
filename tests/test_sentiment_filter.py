import os
from crypto_bot.sentiment_filter import too_bearish, boost_factor


def test_too_bearish(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "50")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "90")
    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        lambda symbol: 30,
    )
    assert too_bearish(20, 40, symbol="BTC") is True


def test_boost_factor(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "90")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "10")
    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        lambda symbol: 80,
    )
    assert boost_factor(70, 60, symbol="BTC") > 1.0


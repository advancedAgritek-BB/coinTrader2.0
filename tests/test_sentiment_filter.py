import os
from crypto_bot.sentiment_filter import too_bearish, boost_factor


def test_too_bearish(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "10")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "30")
    assert too_bearish(20, 40) is True


def test_boost_factor(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "80")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "75")
    assert boost_factor(70, 60) > 1.0


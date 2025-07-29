from crypto_bot.sentiment_filter import too_bearish, boost_factor


def test_too_bearish(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        lambda symbol: 30,
    )
    assert too_bearish(40, symbol="BTC") is True


def test_boost_factor(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.sentiment_filter.lunar_client.get_sentiment",
        lambda symbol: 80,
    )
    assert boost_factor(60, symbol="BTC") > 1.0


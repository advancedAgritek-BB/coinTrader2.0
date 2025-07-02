import pandas as pd
from crypto_bot.volatility_filter import too_flat, too_hot, fetch_funding_rate


def test_too_flat_returns_true():
    data = {
        "high": [1]*15,
        "low": [1]*15,
        "close": [1]*15,
    }
    df = pd.DataFrame(data)
    assert too_flat(df, 0.01) is True


def test_too_hot(monkeypatch):
    monkeypatch.setenv("MOCK_FUNDING_RATE", "0.06")
    assert too_hot("BTCUSDT", 0.05) is True


def test_funding_url_env(monkeypatch):
    base = (
        "https://futures.kraken.com/derivatives/api/v3/"
        "historical-funding-rates?symbol="
    )
    monkeypatch.setenv("FUNDING_RATE_URL", base)
    called = {}

    def fake_get(url, timeout=5):
        called["url"] = url

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"rates": [{"relativeFundingRate": 0.01}, {"relativeFundingRate": 0.02}]}

        return FakeResp()

    monkeypatch.setattr("crypto_bot.volatility_filter.requests.get", fake_get)
    rate = fetch_funding_rate("ETHUSD")
    assert rate == 0.02
    assert called["url"] == base + "ETHUSD"


def test_fetch_funding_rate_symbol_param(monkeypatch):
    monkeypatch.setenv("FUNDING_RATE_URL", "https://api.example.com?symbol=")
    called = {}

    def fake_get(url, timeout=5):
        called["url"] = url

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"rates": [{"timestamp": "1", "relativeFundingRate": 0.01}]}

        return Resp()

    monkeypatch.setattr("crypto_bot.volatility_filter.requests.get", fake_get)
    rate = fetch_funding_rate("ETHUSD")
    assert rate == 0.01
    assert called["url"] == "https://api.example.com?symbol=ETHUSD"



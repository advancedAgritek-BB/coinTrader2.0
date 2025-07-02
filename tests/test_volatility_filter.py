import pandas as pd
from crypto_bot.volatility_filter import too_flat, too_hot


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
    monkeypatch.setenv("FUNDING_RATE_URL", "https://example.com")
    called = {}

    def fake_get(url, timeout=5):
        called["url"] = url

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"result": {"ETHUSD": {"fr": 0.01}}}

        return FakeResp()

    monkeypatch.setattr("crypto_bot.volatility_filter.requests.get", fake_get)
    assert too_hot("ETHUSD", 0.1) is False
    assert called["url"] == "https://example.com/ETHUSD"
    assert called["url"] == "https://example.com?pair=ETHUSD"



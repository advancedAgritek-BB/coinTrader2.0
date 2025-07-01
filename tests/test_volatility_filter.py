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



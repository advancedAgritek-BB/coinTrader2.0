import pandas as pd

from crypto_bot.regime.regime_classifier import classify_regime


def test_classify_regime_returns_unknown_for_short_df():
    data = {
        "open": list(range(10)),
        "high": list(range(1, 11)),
        "low": list(range(10)),
        "close": list(range(10)),
        "volume": [100] * 10,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df) == "unknown"


def test_classify_regime_handles_index_error(monkeypatch):
    data = {
        "open": list(range(30)),
        "high": list(range(1, 31)),
        "low": list(range(30)),
        "close": list(range(30)),
        "volume": [100] * 30,
    }
    df = pd.DataFrame(data)

    def raise_index(*args, **kwargs):
        raise IndexError

    monkeypatch.setattr(
        __import__("ta").trend, "adx", raise_index
    )

    assert classify_regime(df) == "unknown"

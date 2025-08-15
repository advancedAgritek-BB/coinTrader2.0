import pandas as pd
from crypto_bot.volatility_filter import atr_pct, too_flat


def _dummy_df() -> pd.DataFrame:
    data = {
        "high": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "low": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "close": [1] * 15,
    }
    return pd.DataFrame(data)


def test_atr_pct(monkeypatch):
    def fake_atr(df, period=14):
        return pd.Series([0.01] * len(df))

    monkeypatch.setattr("crypto_bot.volatility_filter._calc_atr", fake_atr)
    result = atr_pct(_dummy_df(), period=14)
    assert float(result.iloc[-1]) == 0.01


def test_too_flat(monkeypatch):
    def fake_atr(df, period=14):
        return df["close"] * 0.003  # 0.3%

    monkeypatch.setattr("crypto_bot.volatility_filter._calc_atr", fake_atr)
    df = _dummy_df()
    assert too_flat(df, atr_period=14, threshold=0.004) is True
    # now higher volatility
    monkeypatch.setattr(
        "crypto_bot.volatility_filter._calc_atr",
        lambda d, period=14: d["close"] * 0.01,
    )
    assert too_flat(df, atr_period=14, threshold=0.004) is False

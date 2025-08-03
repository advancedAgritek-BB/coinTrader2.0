import pandas as pd
import crypto_bot.volatility as vol


def _dummy_df():
    return pd.DataFrame({
        "high": [1, 2, 3, 4, 5],
        "low": [0, 1, 2, 3, 4],
        "close": [1, 2, 3, 4, 5],
    })


def test_default_windows(monkeypatch):
    calls = []

    def fake_atr(df, window):
        calls.append(window)
        return 1.0

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    vol.normalize_score_by_volatility(_dummy_df(), 1.0)
    assert calls == [5, 20]


def test_multiplier_cap(monkeypatch):
    def fake_atr(df, window):
        return 100.0 if window == 5 else 1.0

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 1.0)
    assert result == 2.0

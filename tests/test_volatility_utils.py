
import pandas as pd
import pytest
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


@pytest.mark.parametrize(
    "current_atr,long_term_atr",
    [
        (float("nan"), 1.0),
        (1.0, float("nan")),
    ],
)
def test_nan_atr_returns_raw_score(monkeypatch, current_atr, long_term_atr):
    def fake_atr(df, window):
        return current_atr if window == 5 else long_term_atr

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 1.0)
    assert result == 1.0

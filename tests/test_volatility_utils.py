import pandas as pd
import pytest
import crypto_bot.utils.volatility as vol


def _dummy_df():
    return pd.DataFrame(
        {
            "high": [1, 2, 3, 4, 5],
            "low": [0, 1, 2, 3, 4],
            "close": [1, 2, 3, 4, 5],
        }
    )


def test_normalize_low_high(monkeypatch):
    def fake_atr_low(df, period=14):
        return df["close"] * 0.001

    def fake_atr_high(df, period=14):
        return df["close"] * 0.03

    df = _dummy_df()
    monkeypatch.setattr(vol, "calc_atr", fake_atr_low)
    assert vol.normalize_score_by_volatility(1.0, df) == pytest.approx(0.25)
    monkeypatch.setattr(vol, "calc_atr", fake_atr_high)
    assert vol.normalize_score_by_volatility(1.0, df) == pytest.approx(2.0)


def test_accepts_legacy_order(monkeypatch):
    def fake_atr(df, period=14):
        return df["close"] * 0.01

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    df = _dummy_df()
    assert vol.normalize_score_by_volatility(df, 1.0) == pytest.approx(1.0)

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


def test_normalize_uses_window(monkeypatch):
    captured = {}

    def fake_atr(df, period=14, as_series=False):
        captured["val"] = period
        return 2.0

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 14.0, atr_period=7)
    assert captured["val"] == 7
    assert result == pytest.approx(7.0)


def test_returns_input_when_atr_nan(monkeypatch):
    monkeypatch.setattr(vol, "calc_atr", lambda *a, **k: float("nan"))
    df = _dummy_df()
    assert vol.normalize_score_by_volatility(df, 5.0) == pytest.approx(5.0)

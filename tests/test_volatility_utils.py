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


@pytest.mark.parametrize("pname", ["period", "window", "length", "n"])
def test_calc_atr_keyword_params(monkeypatch, pname):
    captured = {}
    if pname == "period":
        def fake_atr(df, period):
            captured["val"] = period
            return pd.Series([2.0])
    elif pname == "window":
        def fake_atr(df, window):
            captured["val"] = window
            return pd.Series([2.0])
    elif pname == "length":
        def fake_atr(df, length):
            captured["val"] = length
            return pd.Series([2.0])
    else:  # pname == "n"
        def fake_atr(df, n):
            captured["val"] = n
            return pd.Series([2.0])

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 14.0, atr_period=7)
    assert captured["val"] == 7
    assert result == pytest.approx(7.0)


def test_calc_atr_positional(monkeypatch):
    captured = {}

    def fake_atr(df, foo):
        captured["val"] = foo
        return pd.Series([2.0])

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 14.0, atr_period=7)
    assert captured["val"] == 7
    assert result == pytest.approx(7.0)


def test_calc_atr_df_only(monkeypatch):
    called = {}

    def fake_atr(df):
        called["yes"] = True
        return pd.Series([2.0])

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    result = vol.normalize_score_by_volatility(_dummy_df(), 14.0, atr_period=7)
    assert called.get("yes")
    assert result == pytest.approx(7.0)


def test_returns_score_on_failure(monkeypatch):
    def fake_atr(df, period=14):
        raise RuntimeError("bad atr")

    monkeypatch.setattr(vol, "calc_atr", fake_atr)
    df = _dummy_df()
    assert vol.normalize_score_by_volatility(df, 5.0) == pytest.approx(5.0)


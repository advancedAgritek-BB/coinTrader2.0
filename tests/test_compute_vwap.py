import pytest
pytest.importorskip("pandas")
import pandas as pd
from crypto_bot.strategy.grid_bot import compute_vwap


def test_compute_vwap_basic():
    df = pd.DataFrame({
        "high": [1, 2, 3, 4],
        "low": [0, 1, 2, 3],
        "close": [1, 2, 3, 4],
        "volume": [1, 1, 1, 1],
    })
    result = compute_vwap(df, 2)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    vol_sum = df["volume"].rolling(2).sum()
    price_sum = pv.rolling(2).sum()
    expected = price_sum / vol_sum
    expected = expected.fillna(typical.rolling(2, min_periods=1).mean())
    expected = expected.where(vol_sum != 0, typical.rolling(2, min_periods=1).mean())
    pd.testing.assert_series_equal(result, expected)


def test_compute_vwap_zero_volume_fallback():
    df = pd.DataFrame({
        "high": [1, 2, 3, 4],
        "low": [0, 1, 2, 3],
        "close": [1, 2, 3, 4],
        "volume": [0, 0, 0, 0],
    })
    result = compute_vwap(df, 3)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    expected = typical.rolling(3, min_periods=1).mean()
    pd.testing.assert_series_equal(result, expected)

import pandas as pd
from crypto_bot.utils.stats import zscore


def test_zscore_values():
    s = pd.Series([1, 2, 3, 4, 5])
    result = zscore(s, lookback=3)
    expected = pd.Series([-3.0, -2.0, -1.0, 0.0, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_zscore_insufficient():
    s = pd.Series([1, 2])
    result = zscore(s, lookback=3)
    assert result.empty

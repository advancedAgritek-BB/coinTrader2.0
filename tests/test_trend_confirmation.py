import pandas as pd
from crypto_bot.utils.trend_confirmation import confirm_multi_tf_trend


def _make_trend_df(rows=60):
    close = pd.Series(range(rows))
    df = pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": [10] * rows,
    })
    return df


def test_confirm_multi_tf_trend_true():
    low = _make_trend_df()
    high = _make_trend_df()
    assert confirm_multi_tf_trend(low, high) is True


def test_confirm_multi_tf_trend_false_when_high_missing():
    low = _make_trend_df()
    high = pd.DataFrame()
    assert confirm_multi_tf_trend(low, high) is False

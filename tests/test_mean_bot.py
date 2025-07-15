import pandas as pd
import pytest

import crypto_bot.strategy.mean_bot as mean_bot


def _df_with_drop(price: float, last_width: float = 5.0) -> pd.DataFrame:
    """Return a dataframe with a final candle width of ``last_width``.

    The preceding candles have a constant width of ``2`` which becomes the
    bandwidth median. Passing a ``last_width`` larger than ``2`` simulates a
    high bandwidth environment while values below ``2`` create a squeeze.
    """

    base = [100.0] * 55 + [price]
    highs = [p + 1 for p in base]
    lows = [p - 1 for p in base]
    highs[-1] = price + last_width / 2
    lows[-1] = price - last_width / 2

    data = {
        "open": base,
        "high": highs,
        "low": lows,
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def _df_low_bw_drop() -> pd.DataFrame:
    import numpy as np

    np.random.seed(0)
    base = list(100 + np.random.randn(60) * 10)
    for i in range(40, 56):
        base[i] = 100 + np.random.randn() * 2
    base = base[:55] + [base[54] - 2]

    data = {
        "open": base,
        "high": [p + 1 for p in base],
        "low": [p - 1 for p in base],
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def test_long_signal_on_big_drop():
    """High bandwidth should block the signal."""
    df = _df_with_drop(80.0)
    score, direction = mean_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_short_signal_on_big_spike():
    """High bandwidth should block the signal."""
    df = _df_with_drop(120.0)
    score, direction = mean_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


@pytest.mark.parametrize("price,expected", [(80.0, "long"), (120.0, "short")])
def test_signal_during_squeeze(price: float, expected: str):
    """A drop or spike during a squeeze should still generate a signal."""
    df = _df_with_drop(price, last_width=1.0)
    score, direction = mean_bot.generate_signal(df)
    assert direction == expected
def test_long_signal_when_bandwidth_low():
    df = _df_low_bw_drop()
    score, direction = mean_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_no_signal_when_trending():
    close = list(range(1, 31))
    df = pd.DataFrame({
        "open": close,
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [100] * len(close),
    })
    score, direction = mean_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0



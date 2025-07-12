import pandas as pd
from crypto_bot.strategy import dca_bot


def _df(close_last: float) -> pd.DataFrame:
    close = [100.0] * 19 + [close_last]
    return pd.DataFrame({"close": close})


def test_long_signal_when_below_ma():
    df = _df(90.0)
    score, direction = dca_bot.generate_signal(df)
    assert direction == "long"
    assert score == 0.8


def test_no_signal_when_above_ma():
    df = _df(100.0)
    score, direction = dca_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0

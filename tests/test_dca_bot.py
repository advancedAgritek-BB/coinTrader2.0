from crypto_bot.strategy import dca_bot
import pandas as pd


def _df(value: float) -> pd.DataFrame:
    data = {"close": [100.0] * 19 + [value]}
    return pd.DataFrame(data)


def test_long_signal_when_below_ma():
    df = _df(89.0)
    score, direction = dca_bot.generate_signal(df)
    assert direction == "long"
    assert score == 0.8


def test_no_signal_above_ma():
    df = _df(90.0)
    score, direction = dca_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_empty_dataframe():
    score, direction = dca_bot.generate_signal(pd.DataFrame())
    assert direction == "none"
    assert score == 0.0

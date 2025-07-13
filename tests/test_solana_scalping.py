import pandas as pd
from crypto_bot.solana import scalping


def make_df(prices):
    return pd.DataFrame({"close": prices})


def test_scalping_long_signal():
    prices = list(range(100, 40, -1)) + [41, 42, 43]
    df = make_df(prices)
    score, direction = scalping.generate_signal(df)
    assert direction == "long"
    assert 0 < score <= 1


def test_scalping_short_signal():
    prices = list(range(1, 60)) + [61, 62, 63, 64, 63, 62, 61]
    df = make_df(prices)
    score, direction = scalping.generate_signal(df)
    assert direction == "short"
    assert 0 < score <= 1


def test_scalping_neutral_signal():
    df = make_df([50] * 40)
    score, direction = scalping.generate_signal(df)
    assert (score, direction) == (0.0, "none")

import pandas as pd

import pytest

from crypto_bot.strategy import breakout_bot


def _make_df(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )


@pytest.fixture
def breakout_df():
    """Factory for a squeeze period followed by a breakout candle."""

    def _factory(direction="long", volume_spike=True, breakout=True):
        base = 100
        prices = [base] * 25
        if direction == "long":
            last_price = base + (3 if breakout else 1)
        else:
            last_price = base - (3 if breakout else 1)
        prices.append(last_price)

        volumes = [100] * 25 + ([300] if volume_spike else [100])
        return _make_df(prices, volumes)

    return _factory


@pytest.fixture
def higher_squeeze_df():
    prices = [100] * 21
    volumes = [100] * 21
    return _make_df(prices, volumes)


@pytest.fixture
def no_squeeze_df():
    prices = list(range(80, 106))
    volumes = [100] * 26
    return _make_df(prices, volumes)


def test_long_breakout_signal():
    prices = [100] * 25 + [103]
    volumes = [100] * 25 + [300]
    df = _make_df(prices, volumes)
    score, direction = breakout_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_short_breakout_signal():
    prices = [100] * 25 + [97]
    volumes = [100] * 25 + [300]
    df = _make_df(prices, volumes)
    score, direction = breakout_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0


def test_requires_squeeze():
    prices = list(range(80, 106))
    volumes = [100] * 26
    df = _make_df(prices, volumes)
    score, direction = breakout_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


@pytest.mark.parametrize("direction", ["long", "short"])
def test_signal_requires_all_conditions(direction, breakout_df, higher_squeeze_df, no_squeeze_df):
    df = breakout_df(direction)
    score, got = breakout_bot.generate_signal(df, higher_df=higher_squeeze_df)
    assert got == direction and score > 0

    df_no_vol = breakout_df(direction, volume_spike=False)
    assert breakout_bot.generate_signal(df_no_vol, higher_df=higher_squeeze_df)[1] == "none"

    df_no_break = breakout_df(direction, breakout=False)
    assert breakout_bot.generate_signal(df_no_break, higher_df=higher_squeeze_df)[1] == "none"

    assert breakout_bot.generate_signal(no_squeeze_df, higher_df=higher_squeeze_df)[1] == "none"


@pytest.mark.parametrize("direction", ["long", "short"])
def test_higher_timeframe_squeeze_required(direction, breakout_df, higher_squeeze_df, no_squeeze_df):
    df = breakout_df(direction)
    _, got = breakout_bot.generate_signal(df, higher_df=higher_squeeze_df)
    assert got == direction

    _, got_none = breakout_bot.generate_signal(df, higher_df=no_squeeze_df)
    assert got_none == "none"

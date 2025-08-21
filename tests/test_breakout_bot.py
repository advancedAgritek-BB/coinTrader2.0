import pandas as pd

import pytest
import types
import pandas as pd

from crypto_bot.strategy import breakout_bot
from crypto_bot.utils import indicator_cache, volatility
from crypto_bot.utils.volatility import normalize_score_by_volatility

BASE_CFG = {"breakout": {"ema_window": 5, "adx_window": 3, "adx_threshold": 0}}


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
        prices = [base] * 35
        if direction == "long":
            last_price = base + (2 if breakout else 1)
        else:
            last_price = base - (2 if breakout else 1)
        prices.append(last_price)

        volumes = [100] * 35 + ([300] if volume_spike else [100])
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


def test_handles_insufficient_history():
    indicator_cache.CACHE.clear()
    cfg = {
        "breakout": {
            "bb_length": 20,
            "kc_length": 20,
            "dc_length": 5,
            "volume_window": 5,
        }
    }
    prices = [100] * 20
    volumes = [100] * 20
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df, config=cfg)
    assert direction == "none" and score == 0.0
    indicator_cache.CACHE.clear()


def test_long_breakout_signal():
    prices = [100] * 35 + [102]
    volumes = [100] * 35 + [300]
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df, config=BASE_CFG)
    assert direction == "long"
    assert score > 0


def test_short_breakout_signal():
    prices = [100] * 35 + [98]
    volumes = [100] * 35 + [300]
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df, config=BASE_CFG)
    assert direction == "short"
    assert score < 0


def test_requires_squeeze():
    prices = list(range(80, 106))
    volumes = [100] * 26
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df, config=BASE_CFG)
    assert direction == "none"
    assert score == 0.0


@pytest.mark.parametrize("direction", ["long", "short"])
def test_signal_requires_all_conditions(direction, breakout_df, higher_squeeze_df, no_squeeze_df):
    df = breakout_df(direction)
    score, got = breakout_bot.generate_signal(df, config=BASE_CFG, higher_df=higher_squeeze_df)
    assert got == direction and (score > 0 if direction == "long" else score < 0)

    df_no_vol = breakout_df(direction, volume_spike=False)
    score_no_vol, got_no_vol = breakout_bot.generate_signal(
        df_no_vol, BASE_CFG, higher_df=higher_squeeze_df
    )
    assert got_no_vol == direction and (
        score_no_vol > 0 if direction == "long" else score_no_vol < 0
    )

    df_no_break = breakout_df(direction, breakout=False)
    assert (
        breakout_bot.generate_signal(df_no_break, config=BASE_CFG, higher_df=higher_squeeze_df)[1]
        == "none"
    )

    assert (
        breakout_bot.generate_signal(no_squeeze_df, config=BASE_CFG, higher_df=higher_squeeze_df)[1]
        == "none"
    )


@pytest.mark.parametrize("direction", ["long", "short"])
def test_higher_timeframe_optional(direction, breakout_df, higher_squeeze_df, no_squeeze_df):
    df = breakout_df(direction)
    _, got = breakout_bot.generate_signal(df, config=BASE_CFG, higher_df=higher_squeeze_df)
    assert got == direction

    _, got_no = breakout_bot.generate_signal(df, config=BASE_CFG, higher_df=no_squeeze_df)
    assert got_no == direction


def test_squeeze_zscore(monkeypatch):
    prices = [100] * 15 + [103]
    volumes = [100] * 15 + [300]
    df = _make_df(prices, volumes)

    monkeypatch.setattr(
        breakout_bot.stats,
        "zscore",
        lambda s, lookback=3: pd.Series([-2] * len(s), index=s.index),
    )

    cfg = {
        "indicator_lookback": 3,
        "bb_squeeze_pct": 20,
        "breakout": {
            "bb_length": 3,
            "kc_length": 3,
            "dc_length": 3,
            "volume_window": 3,
            "ema_window": 5,
            "adx_window": 3,
            "adx_threshold": 0,
        },
    }
    score, direction, _ = breakout_bot.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0


def test_no_volume_confirmation_still_signals(breakout_df):
    df = breakout_df("long", volume_spike=False)
    cfg = {
        "breakout": {
            "vol_confirmation": False,
            "ema_window": 5,
            "adx_window": 3,
            "adx_threshold": 0,
        }
    }
    score, direction, _ = breakout_bot.generate_signal(df, config=cfg)
    assert direction == "long" and score > 0


def test_trainer_model_influence(monkeypatch):
    prices = [100] * 35 + [102]
    volumes = [100] * 35 + [300]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": volumes,
    })
    monkeypatch.setattr(breakout_bot, "MODEL", None)
    raw_base, direction, _ = breakout_bot.generate_signal(
        df, {"atr_normalization": False}
    )
    dummy = types.SimpleNamespace(predict=lambda _df: 0.5)
    monkeypatch.setattr(breakout_bot, "MODEL", dummy)
    score, direction2, _ = breakout_bot.generate_signal(df)
    recent = df.iloc[-31:]
    expected = volatility.normalize_score_by_volatility(
        recent, (raw_base + 0.5) / 2
    )
    assert direction2 == direction
    base_raw_cfg = dict(BASE_CFG)
    base_raw_cfg["atr_normalization"] = False
    base_raw, direction, _ = breakout_bot.generate_signal(df, config=base_raw_cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.5)
    monkeypatch.setattr(breakout_bot, "MODEL", dummy)
    score, direction2, _ = breakout_bot.generate_signal(df, config=BASE_CFG)
    assert direction2 == direction
    recent = df.iloc[-31:]
    expected = normalize_score_by_volatility(recent, (base_raw + 0.5) / 2)
    assert score == pytest.approx(expected, rel=1e-5)

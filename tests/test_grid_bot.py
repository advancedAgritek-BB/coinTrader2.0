import pandas as pd
import pytest
import sys
import types

sys.modules['crypto_bot.strategy.sniper_bot'] = types.ModuleType('sniper_bot')
from crypto_bot.strategy import grid_bot
from crypto_bot.strategy.grid_bot import GridConfig
from crypto_bot import grid_state


def _df_with_price(price: float, volume: float = 300.0) -> pd.DataFrame:
    data = {
        "high": [110.0] * 20,
        "low": [90.0] * 20,
        "close": [100.0] * 19 + [price],
        "volume": [100.0] * 19 + [volume],
    }
    return pd.DataFrame(data)


def _df_trend_down() -> pd.DataFrame:
    close = list(range(110, 91, -1)) + [89]
    data = {
        "high": [110.0] * 20,
        "low": [90.0] * 20,
        "close": close,
    }
    return pd.DataFrame(data)


def _df_trend_up() -> pd.DataFrame:
    close = list(range(90, 109)) + [111]
    data = {
        "high": [110.0] * 20,
        "low": [90.0] * 20,
        "close": close,
    }
    return pd.DataFrame(data)


def _df_range_change() -> pd.DataFrame:
    data = {
        "high": [150.0] * 20 + [110.0] * 10,
        "low": [50.0] * 20 + [90.0] * 10,
        "close": [100.0] * 29 + [89.0],
    }
    return pd.DataFrame(data)


def _df_small_range(price: float = 100.0) -> pd.DataFrame:
    data = {
        "high": [100.5] * 20,
        "low": [99.5] * 20,
        "close": [100.0] * 19 + [price],
        "volume": [100.0] * 19 + [300.0],
    }
    return pd.DataFrame(data)


def test_short_signal_above_upper_grid(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 1.0)
    df = _df_with_price(111.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "short"
    assert score > 0.9


def test_long_signal_below_lower_grid(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 1.0)
    df = _df_with_price(89.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "long"
    assert score > 0.9


def test_no_signal_in_middle(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 1.0)
    df = _df_with_price(102.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "none"
    assert score == 0.0


def test_grid_levels_env_override(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    df = _df_with_price(102.0)
    _, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "none"

    monkeypatch.setenv("GRID_LEVELS", "3")
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "short"
    assert score > 0.0


def test_breakout_trigger_short(monkeypatch):
    df = _df_with_price(120.0)

    called = {}

    def fake_breakout_signal(dataframe, config=None):
        called['called'] = True
        return 0.6, "short"

    monkeypatch.setattr(grid_bot.breakout_bot, "generate_signal", fake_breakout_signal)
    score, direction = grid_bot.generate_signal(df)
    assert called.get('called')
    assert direction == "short"
    assert score == 0.6


def test_breakout_trigger_long(monkeypatch):
    df = _df_with_price(80.0)

    called = {}

    def fake_breakout_signal(dataframe, config=None):
        called['called'] = True
        return 0.7, "long"

    monkeypatch.setattr(grid_bot.breakout_bot, "generate_signal", fake_breakout_signal)
    score, direction = grid_bot.generate_signal(df)
    assert called.get('called')
    assert direction == "long"
    assert score == 0.7


def test_cooldown_blocks_signal(monkeypatch):
    df = _df_with_price(89.0)
    cfg = GridConfig(symbol="XBT/USDT")
    grid_state.clear()
    monkeypatch.setattr(grid_bot.grid_state, "in_cooldown", lambda s, b: True)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert (score, direction) == (0.0, "none")


def test_volume_filter_blocks_short_signal():
    df = _df_with_price(111.0, volume=100.0)
    score, direction = grid_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_respects_active_leg_limit(monkeypatch):
    df = _df_with_price(89.0)
    cfg = GridConfig(symbol="XBT/USDT")
    grid_state.clear()
    monkeypatch.setattr(grid_bot.grid_state, "active_leg_count", lambda s: 5)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert direction == "none"
    assert score == 0.0


def test_volume_filter_blocks_long_signal():
    df = _df_with_price(89.0, volume=100.0)
    score, direction = grid_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_atr_spacing(monkeypatch):
    df = _df_with_price(106.0)
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 2.0)
    narrow, _ = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 10.0)
    wide, _ = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert narrow > wide


def test_small_range_no_signal(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([0.5]))
    df = _df_small_range()
    cfg = GridConfig(min_range_pct=0.02, atr_normalization=False)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert (score, direction) == (0.0, "none")


def test_dynamic_grid_spacing_persists(monkeypatch):
    atr_vals = iter([1.0, 1.1])
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: next(atr_vals))
    grid_state.clear()
    cfg = GridConfig(symbol="XBT/USDT", dynamic_grid=True, atr_normalization=False)
    df = _df_with_price(111.0)
    grid_bot.generate_signal(df, config=cfg)
    first = grid_state.get_grid_step("XBT/USDT")
    df2 = _df_with_price(111.0)
    grid_bot.generate_signal(df2, config=cfg)
    second = grid_state.get_grid_step("XBT/USDT")
    assert first == second


def test_grid_step_realigns_on_large_atr_change(monkeypatch):
    atr_vals = iter([1.0, 2.0])
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: next(atr_vals))
    grid_state.clear()
    cfg = GridConfig(symbol="XBT/USDT", dynamic_grid=True, atr_normalization=False)
    df = _df_with_price(111.0)
    grid_bot.generate_signal(df, config=cfg)
    first = grid_state.get_grid_step("XBT/USDT")
    df2 = _df_with_price(111.0)
    grid_bot.generate_signal(df2, config=cfg)
    second = grid_state.get_grid_step("XBT/USDT")
    assert second > first


def test_long_blocked_by_trend_filter():
    df = _df_trend_down()
    cfg = {"trend_ema_fast": 3, "trend_ema_slow": 5}
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert (score, direction) == (0.0, "none")


def test_short_blocked_by_trend_filter():
    df = _df_trend_up()
    cfg = {"trend_ema_fast": 3, "trend_ema_slow": 5}
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert (score, direction) == (0.0, "none")


@pytest.mark.parametrize("cfg", [{"range_window": 10}, GridConfig(range_window=10)])
def test_range_window_config(cfg, monkeypatch):
    df = _df_range_change()
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 18.0)
    _, default_direction = grid_bot.generate_signal(df)
    assert default_direction == "none"

    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 1.0)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0.0


def test_trainer_model_influence(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, period=14: pd.Series([5.0]))
    monkeypatch.setattr(grid_bot, "atr_percent", lambda df, window=14: 1.0)
    df = _df_with_price(111.0)
    cfg = GridConfig(atr_normalization=False)
    monkeypatch.setattr(grid_bot, "MODEL", None)
    base, direction = grid_bot.generate_signal(df, config=cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.4)
    monkeypatch.setattr(grid_bot, "MODEL", dummy)
    score, direction2 = grid_bot.generate_signal(df, config=cfg)
    assert direction2 == direction
    assert score == pytest.approx((base + 0.4) / 2)


def test_zero_price_returns_none():
    df = _df_with_price(0.0)
    score, direction = grid_bot.generate_signal(df)
    assert (score, direction) == (0.0, "none")

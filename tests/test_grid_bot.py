import pandas as pd
import pytest
from crypto_bot.strategy import grid_bot
from crypto_bot.strategy.grid_bot import GridConfig


def _df_with_price(price: float) -> pd.DataFrame:
    data = {
        "high": [110.0] * 20,
        "low": [90.0] * 20,
        "close": [100.0] * 19 + [price],
    }
    return pd.DataFrame(data)


def test_short_signal_above_upper_grid(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 5.0)
def _df_range_change() -> pd.DataFrame:
    data = {
        "high": [150.0] * 20 + [110.0] * 10,
        "low": [50.0] * 20 + [90.0] * 10,
        "close": [100.0] * 29 + [89.0],
    }
    return pd.DataFrame(data)


def test_short_signal_above_upper_grid():
    df = _df_with_price(111.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "short"
    assert score > 0.9


def test_long_signal_below_lower_grid(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 5.0)
    df = _df_with_price(89.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "long"
    assert score > 0.9


def test_no_signal_in_middle(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 5.0)
    df = _df_with_price(102.0)
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "none"
    assert score == 0.0


def test_grid_levels_env_override(monkeypatch):
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 5.0)
    df = _df_with_price(102.0)
    # default configuration -> neutral
    _, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "none"

    monkeypatch.setenv("GRID_LEVELS", "3")
    score, direction = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert direction == "short"
    assert score > 0.0


def test_atr_spacing(monkeypatch):
    df = _df_with_price(106.0)
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 2.0)
    narrow, _ = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    monkeypatch.setattr(grid_bot, "calc_atr", lambda df, window=14: 10.0)
    wide, _ = grid_bot.generate_signal(df, config=GridConfig(atr_normalization=False))
    assert narrow > wide

@pytest.mark.parametrize("cfg", [{"range_window": 10}, GridConfig(range_window=10)])
def test_range_window_config(cfg):
    df = _df_range_change()
    _, default_direction = grid_bot.generate_signal(df)
    assert default_direction == "none"

    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0.0

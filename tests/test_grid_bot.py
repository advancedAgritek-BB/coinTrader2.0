import pandas as pd
from crypto_bot.strategy import grid_bot
from crypto_bot.strategy.grid_bot import GridConfig
from crypto_bot import grid_state


def _df_with_price(price: float) -> pd.DataFrame:
    data = {
        "high": [110.0] * 20,
        "low": [90.0] * 20,
        "close": [100.0] * 19 + [price],
    }
    return pd.DataFrame(data)


def test_short_signal_above_upper_grid():
    df = _df_with_price(111.0)
    score, direction = grid_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0.9


def test_long_signal_below_lower_grid():
    df = _df_with_price(89.0)
    score, direction = grid_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.9


def test_no_signal_in_middle():
    df = _df_with_price(102.0)
    score, direction = grid_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_grid_levels_env_override(monkeypatch):
    df = _df_with_price(102.0)
    # default configuration -> neutral
    _, direction = grid_bot.generate_signal(df)
    assert direction == "none"

    monkeypatch.setenv("GRID_LEVELS", "3")
    score, direction = grid_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0.0


def test_cooldown_blocks_signal(monkeypatch):
    df = _df_with_price(89.0)
    cfg = GridConfig(symbol="BTC/USDT")
    grid_state.clear()
    monkeypatch.setattr(grid_bot.grid_state, "in_cooldown", lambda s, b: True)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert direction == "none"
    assert score == 0.0


def test_respects_active_leg_limit(monkeypatch):
    df = _df_with_price(89.0)
    cfg = GridConfig(symbol="BTC/USDT")
    grid_state.clear()
    monkeypatch.setattr(grid_bot.grid_state, "active_leg_count", lambda s: 5)
    score, direction = grid_bot.generate_signal(df, config=cfg)
    assert direction == "none"
    assert score == 0.0

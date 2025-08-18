import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "momentum_bot",
    Path(__file__).resolve().parents[1]
    / "crypto_bot"
    / "strategy"
    / "momentum_bot.py",
)
momentum_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(momentum_bot)


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
    """Factory producing a Donchian breakout with optional volume spike."""

    def _factory(volume_spike=True, breakout=True):
        base = 100
        prices = [base] * 25
        prices.append(base + 2 if breakout else base)
        volumes = [100] * 25 + ([300] if volume_spike else [100])
        return _make_df(prices, volumes)

    return _factory


def test_breakout_with_volume_spike(breakout_df):
    df = breakout_df()
    score, direction = momentum_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_no_signal_missing_conditions(breakout_df):
    df_no_breakout = breakout_df(breakout=False)
    score, direction = momentum_bot.generate_signal(df_no_breakout)
    assert direction == "long"
    assert score > 0

    df_no_volume = breakout_df(volume_spike=False)
    score, direction = momentum_bot.generate_signal(df_no_volume)
    assert direction == "long"
    assert score > 0

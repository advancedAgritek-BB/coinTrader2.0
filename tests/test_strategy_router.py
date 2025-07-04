import pytest

from crypto_bot.strategy_router import strategy_for
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    mean_bot,
    breakout_bot,
    sniper_bot,
    micro_scalp_bot,
)


def test_strategy_for_mapping():
    assert strategy_for("trending") is trend_bot.generate_signal
    assert strategy_for("sideways") is grid_bot.generate_signal
    assert strategy_for("mean-reverting") is mean_bot.generate_signal
    assert strategy_for("breakout") is breakout_bot.generate_signal
    assert strategy_for("volatile") is sniper_bot.generate_signal
    assert strategy_for("scalp") is micro_scalp_bot.generate_signal
    assert strategy_for("unknown") is grid_bot.generate_signal

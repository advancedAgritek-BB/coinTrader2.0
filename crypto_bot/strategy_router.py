from typing import Callable
import pandas as pd

from crypto_bot.strategy import trend_bot, grid_bot, sniper_bot, dex_scalper


def route(regime: str, mode: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy function based on regime and environment."""
    if mode == 'cex':
        if regime == 'trending':
            return trend_bot.generate_signal
        return grid_bot.generate_signal
    if mode == 'onchain':
        if regime in {'breakout', 'volatile'}:
            return sniper_bot.generate_signal
        return dex_scalper.generate_signal
    # auto mode defaults
    if regime == 'trending':
        return trend_bot.generate_signal
    if regime in {'breakout', 'volatile'}:
        return sniper_bot.generate_signal
    return grid_bot.generate_signal

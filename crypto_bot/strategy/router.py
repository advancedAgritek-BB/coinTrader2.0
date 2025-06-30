from typing import Callable
import pandas as pd

from crypto_bot.strategy import trend_bot, grid_bot, mean_bot, breakout_bot


def route(regime: str) -> Callable[[pd.DataFrame], tuple]:
    if regime == 'trending':
        return trend_bot.generate_signal
    if regime == 'sideways':
        return grid_bot.generate_signal
    if regime == 'mean-reverting':
        return mean_bot.generate_signal
    if regime == 'breakout':
        return breakout_bot.generate_signal
    return trend_bot.generate_signal

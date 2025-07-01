from typing import Callable
import pandas as pd

import logging

logger = logging.getLogger(__name__)

from crypto_bot.strategy import trend_bot, grid_bot, sniper_bot, dex_scalper


def route(regime: str, mode: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy function based on regime and environment."""
    if mode == 'cex':
        if regime == 'trending':
            logger.info("Routing to trend bot (cex)")
            return trend_bot.generate_signal
        logger.info("Routing to grid bot (cex)")
        return grid_bot.generate_signal
    if mode == 'onchain':
        if regime in {'breakout', 'volatile'}:
            logger.info("Routing to sniper bot (onchain)")
            return sniper_bot.generate_signal
        logger.info("Routing to DEX scalper (onchain)")
        return dex_scalper.generate_signal
    # auto mode defaults
    if regime == 'trending':
        logger.info("Routing to trend bot (auto)")
        return trend_bot.generate_signal
    if regime in {'breakout', 'volatile'}:
        logger.info("Routing to sniper bot (auto)")
        return sniper_bot.generate_signal
    logger.info("Routing to grid bot (auto)")
    return grid_bot.generate_signal

from typing import Callable, Tuple

import pandas as pd

from crypto_bot.utils.logger import setup_logger
from crypto_bot.strategy import trend_bot, grid_bot, sniper_bot, dex_scalper

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def strategy_name(regime: str, mode: str) -> str:
    """Return the name of the strategy for given regime and mode."""
    if mode == "cex":
        return "trend" if regime == "trending" else "grid"
    if mode == "onchain":
        return "sniper" if regime in {"breakout", "volatile"} else "dex_scalper"
    if regime == "trending":
        return "trend"
    if regime in {"breakout", "volatile"}:
        return "sniper"
    return "grid"


def route(
    regime: str, mode: str
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Select a strategy based on market regime and operating mode.

    Parameters
    ----------
    regime : str
        Current market regime as classified by indicators.
    mode : str
        Trading environment, either ``cex``, ``onchain`` or ``auto``.

    Returns
    -------
    Callable[[pd.DataFrame], Tuple[float, str]]
        Strategy function returning a score and trade direction.
    """
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

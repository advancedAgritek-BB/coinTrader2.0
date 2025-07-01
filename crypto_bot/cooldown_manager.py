from __future__ import annotations

from datetime import datetime
from typing import Dict

from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/cooldown.log")

cooldowns: Dict[str, datetime] = {}
MIN_COOLDOWN = 0


def configure(min_cooldown: int) -> None:
    """Set the minimum cooldown in seconds."""
    global MIN_COOLDOWN
    MIN_COOLDOWN = max(0, int(min_cooldown))


def in_cooldown(symbol: str, strategy: str) -> bool:
    """Return ``True`` if the strategy is cooling down for the symbol."""
    key = f"{symbol}_{strategy}"
    if key in cooldowns and (
        datetime.now() - cooldowns[key]
    ).seconds < MIN_COOLDOWN:
        logger.info("%s in cooldown for %s", strategy, symbol)
        return True
    return False


def mark_cooldown(symbol: str, strategy: str) -> None:
    """Record cooldown start time for the strategy on the symbol."""
    cooldowns[f"{symbol}_{strategy}"] = datetime.now()
    logger.info("Marked cooldown for %s on %s", strategy, symbol)

from __future__ import annotations

from datetime import datetime
from typing import Dict

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path
from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace

# Consolidate cooldown logs with the main bot log
logger = setup_logger(__name__, LOG_DIR / "bot.log")

cooldowns: Dict[str, datetime] = {}
MIN_COOLDOWN = 0
_lock = Lock()


def configure(min_cooldown: int) -> None:
    """Set the minimum cooldown in seconds."""
    global MIN_COOLDOWN
    MIN_COOLDOWN = max(0, int(min_cooldown))


def in_cooldown(symbol: str, side: str) -> bool:
    """Return ``True`` if the trade side is cooling down for the symbol."""
    key = f"{symbol}_{side}"
    if key in cooldowns and (
        datetime.now() - cooldowns[key]
    ).seconds < MIN_COOLDOWN:
        logger.info("%s in cooldown for %s", side, symbol)
        return True
    return False


def mark_cooldown(symbol: str, side: str) -> None:
    """Record cooldown start time for the trade side on the symbol."""
    cooldowns[f"{symbol}_{side}"] = datetime.now()
    logger.info("Marked cooldown for %s on %s", side, symbol)


@contextmanager
def cooldown(symbol: str, side: str):
    """Context manager to atomically check and mark cooldown.

    Yields an object with an ``allowed`` attribute indicating whether the
    trade side is currently in cooldown. The object's ``mark`` method will mark
    the cooldown for the symbol and side. Both the check and any subsequent call
    to ``mark`` occur while holding an internal lock, making the operation
    thread-safe.
    """

    if not symbol:
        yield SimpleNamespace(allowed=True, mark=lambda: None)
        return

    with _lock:
        allowed = not in_cooldown(symbol, side)
        ctx = SimpleNamespace(allowed=allowed, mark=lambda: mark_cooldown(symbol, side))
        yield ctx

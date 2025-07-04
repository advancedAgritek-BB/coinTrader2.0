from __future__ import annotations

"""Utilities for sending human-readable trade summaries via Telegram."""

from typing import Optional

from .telegram import send_message
from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def entry_summary(symbol: str, strategy: str, score: float, direction: str) -> str:
    """Return a summary of a trade entry."""
    return (
        f"Entering {direction.upper()} on {symbol} using {strategy}. "
        f"Score: {score:.2f}"
    )


def exit_summary(symbol: str, strategy: str, pnl: float, direction: str) -> str:
    """Return a summary of a trade exit."""
    return (
        f"Exiting {direction.upper()} on {symbol} from {strategy}. "
        f"PnL: {pnl:.2f}"
    )


def report_entry(
    token: str,
    chat_id: str,
    symbol: str,
    strategy: str,
    score: float,
    direction: str,
) -> Optional[str]:
    """Send a Telegram message summarizing a trade entry."""
    err = send_message(token, chat_id, entry_summary(symbol, strategy, score, direction))
    if err:
        logger.error("Failed to report entry: %s", err)
    return err


def report_exit(
    token: str,
    chat_id: str,
    symbol: str,
    strategy: str,
    pnl: float,
    direction: str,
) -> Optional[str]:
    """Send a Telegram message summarizing a trade exit."""
    err = send_message(token, chat_id, exit_summary(symbol, strategy, pnl, direction))
    if err:
        logger.error("Failed to report exit: %s", err)
    return err

from __future__ import annotations

"""Log active trade position and wallet balance."""

from pathlib import Path

from .logger import LOG_DIR, setup_logger

LOG_FILE = LOG_DIR / "positions.log"
logger = setup_logger(__name__, LOG_FILE)


def log_balance(balance: float) -> None:
    """Write the current wallet balance to the log in USD."""
    logger.info("Wallet balance $%.2f", balance)


def log_position(
    symbol: str,
    side: str,
    amount: float,
    entry_price: float,
    current_price: float,
    balance: float,
    pnl: float | None = None,
) -> None:
    """Write a log entry describing the active position.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``"XBT/USDT"``.
    side : str
        ``"buy"`` or ``"sell"``.
    amount : float
        Position size.
    entry_price : float
        Price when the position was opened. Logged with six decimal places.
    current_price : float
        Latest market price. Logged with six decimal places.
    balance : float
        Current wallet balance including unrealized PnL.
    pnl : float, optional
        Realized profit or loss to log instead of computing from prices.
    """
    if pnl is None:
        pnl = (current_price - entry_price) * amount
        if side == "sell":
            pnl = -pnl
    status = "positive" if pnl >= 0 else "negative"
    logger.info(
        "Active %s %s %.4f entry %.6f current %.6f pnl $%.2f (%s) balance $%.2f",
        symbol,
        side,
        amount,
        entry_price,
        current_price,
        pnl,
        status,
        balance,
    )

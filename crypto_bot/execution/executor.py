try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
from typing import Dict
from pathlib import Path
import time

from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def load_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    return exchange


def execute_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    config: Dict,
    notifier: TelegramNotifier,
    dry_run: bool = True,
    max_retries: int = 3,
) -> None:
    """Execute a market trade with optional retry logic.

    Parameters
    ----------
    max_retries:
        Number of attempts when order placement fails due to a transient
        error. Defaults to ``3``.
    """

    pre_msg = f"Placing {side} order for {amount} {symbol}"
    err = notifier.notify(pre_msg)
    if err:
        logger.error("Failed to send message: %s", err)

    if not dry_run:
        for attempt in range(max_retries):
            try:
                order = exchange.create_market_order(symbol, side, amount)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                err_msg = notifier.notify(f"Order failed: {e}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                return
    else:
        order = {'symbol': symbol, 'side': side, 'amount': amount, 'dry_run': True}

    err = notifier.notify(f"Order executed: {order}")
    if err:
        logger.error("Failed to send message: %s", err)
    log_trade(order)

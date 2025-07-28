try:
    import ccxt  # type: ignore
    from ccxt import NetworkError, RateLimitExceeded, ExchangeError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
    NetworkError = RateLimitExceeded = ExchangeError = Exception
from typing import Dict
import time
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
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
        err = notifier.notify(f"Order executed: {order}")
        if err:
            logger.error("Failed to send message: %s", err)
        log_trade(order)
        return

    order = None
    for retry in range(max_retries):
        try:
            order = exchange.create_market_order(symbol, side, amount)
            break
        except (NetworkError, RateLimitExceeded) as exc:
            logger.warning("Transient error executing trade: %s", exc, exc_info=True)
            time.sleep(2 ** retry)
        except ExchangeError as exc:
            logger.error("Exchange error executing trade: %s", exc)
            err_msg = notifier.notify(f"Order failed: {exc}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            raise
        except Exception as exc:
            logger.error("Unexpected error executing trade: %s", exc, exc_info=True)
            err_msg = notifier.notify(f"Order failed: {exc}")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            logger.error(
                "Order failed - symbol=%s side=%s amount=%s: %s",
                symbol,
                side,
                amount,
                e,
                exc_info=True,
            )
            return
            raise
    else:
        logger.error("Order failed after %s retries", max_retries)
        err_msg = notifier.notify("Order failed after retries")
        if err_msg:
            logger.error("Failed to send message: %s", err_msg)
        raise RuntimeError("Order execution failed after retries")

    err = notifier.notify(f"Order executed: {order}")
    if err:
        logger.error("Failed to send message: %s", err)
    log_trade(order)

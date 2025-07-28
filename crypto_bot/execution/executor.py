import os
import time
from typing import TYPE_CHECKING, Any, Dict

import ccxt  # type: ignore
import keyring

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.trade_logger import log_trade

NetworkError = getattr(ccxt, "NetworkError", Exception)
RateLimitExceeded = getattr(ccxt, "RateLimitExceeded", Exception)
ExchangeError = getattr(ccxt, "ExchangeError", Exception)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from crypto_bot.utils.telegram import TelegramNotifier


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def load_exchange(
    api_key: str | None = None, api_secret: str | None = None
) -> 'ccxt.Exchange':
    """Instantiate a Binance exchange using provided or stored credentials."""

    if api_key is None:
        api_key = keyring.get_password("binance", "api_key")
        if api_key is None:
            api_key = os.getenv("API_KEY")

    if api_secret is None:
        api_secret = keyring.get_password("binance", "api_secret")
        if api_secret is None:
            api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Missing Binance API credentials")

    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
    )
    return exchange


def execute_trade(
    exchange: 'ccxt.Exchange',
    symbol: str,
    side: str,
    amount: float,
    config: Dict,
    notifier: Any,
    dry_run: bool = True,
    max_retries: int = 3,
    poll_timeout: int = 60,
) -> Dict:
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

    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
        err = notifier.notify(f"Order executed: {order}")
        if err:
            logger.error("Failed to send message: %s", err)
        log_trade(order)
        return order

    order: Dict | None = None
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
            return {}
    else:
        logger.error("Order failed after %s retries", max_retries)
        err_msg = notifier.notify("Order failed after retries")
        if err_msg:
            logger.error("Failed to send message: %s", err_msg)
        raise RuntimeError("Order execution failed after retries")

    assert order is not None
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= poll_timeout:
            err_msg = notifier.notify("Order polling timed out")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            raise TimeoutError("Order polling timed out")
        try:
            order_status = exchange.fetch_order(order["id"], symbol)
        except Exception as exc:  # pragma: no cover - fetch may fail
            logger.warning("Order status fetch failed: %s", exc)
            time.sleep(1)
            continue

        if order_status.get("status") == "closed":
            filled = float(order_status.get("filled", 0))
            total = float(order_status.get("amount", amount))
            if filled < total:
                remaining = total - filled
                msg = notifier.notify(
                    f"Partial fill detected ({filled}/{total}). Placing remaining {remaining}"
                )
                if msg:
                    logger.error("Failed to send message: %s", msg)
                amount = remaining
                for retry in range(max_retries):
                    try:
                        order = exchange.create_market_order(symbol, side, remaining)
                        start = time.monotonic()
                        break
                    except (NetworkError, RateLimitExceeded):
                        time.sleep(2 ** retry)
                        continue
                else:
                    logger.error("Order failed after %s retries", max_retries)
                    raise RuntimeError("Order execution failed after retries")
                continue

            err = notifier.notify(f"Order executed: {order_status}")
            if err:
                logger.error("Failed to send message: %s", err)
            log_trade(order_status)
            return order_status

        time.sleep(1)

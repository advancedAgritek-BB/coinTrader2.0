import os
import time
import ccxt
import asyncio
from typing import Dict, Optional, Tuple, List

try:
    import ccxt.pro as ccxtpro  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxtpro = None

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot import tax_logger
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/execution.log")


def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client.

    When ``use_websocket`` is enabled and ``ccxtpro`` is available, an
    asynchronous ``ccxt.pro`` instance is returned. Otherwise the standard
    ``ccxt`` exchange is used. ``KrakenWSClient`` is retained for backward
    compatibility when WebSocket trading is desired without ccxt.pro.
    """

    exchange_name = config.get("exchange", "coinbase")
    use_ws = config.get("use_websocket", False)

    ws_client: Optional[KrakenWSClient] = None
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    ws_token = os.getenv("KRAKEN_WS_TOKEN")
    api_token = os.getenv("KRAKEN_API_TOKEN")

    if use_ws and ccxtpro:
        ccxt_mod = ccxtpro
    else:
        ccxt_mod = ccxt

    if exchange_name == "coinbase":
        exchange = ccxt_mod.coinbase(
            {
                "apiKey": os.getenv("API_KEY"),
                "secret": os.getenv("API_SECRET"),
                "password": os.getenv("API_PASSPHRASE"),
                "enableRateLimit": True,
            }
        )
    elif exchange_name == "kraken":
        if use_ws:
            if ccxtpro:
                if (api_key and api_secret) or ws_token:
                    ws_client = KrakenWSClient(api_key, api_secret, ws_token, api_token)
            else:
                ws_client = KrakenWSClient(api_key, api_secret)

        exchange = ccxt_mod.kraken(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    return exchange, ws_client


def execute_trade(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str,
    chat_id: str,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
) -> Dict:
    if use_websocket and ws_client is None and not dry_run:
        raise ValueError("WebSocket trading enabled but ws_client is missing")
    config = config or {}

    def has_liquidity(order_size: float) -> bool:
        try:
            depth = config.get("liquidity_depth", 10)
            ob = exchange.fetch_order_book(symbol, limit=depth)
            book = ob["asks" if side == "buy" else "bids"]
            vol = 0.0
            for _, qty in book:
                vol += qty
                if vol >= order_size:
                    return True
            return False
        except Exception as err:
            send_message(token, chat_id, f"Order book error: {err}")
            return False

    def place(size: float) -> Dict:
        if dry_run:
            return {"symbol": symbol, "side": side, "amount": size, "dry_run": True}
        try:
            if ws_client is not None:
                return ws_client.add_order(symbol, side, size)
            return exchange.create_market_order(symbol, side, size)
        except Exception as exc:
            send_message(token, chat_id, f"Order failed: {exc}")
            return {}

    send_message(token, chat_id, f"Placing {side} order for {amount} {symbol}")

    try:
        ticker = exchange.fetch_ticker(symbol)
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid and ask:
            slippage = (ask - bid) / ((ask + bid) / 2)
            if slippage > config.get("max_slippage_pct", 1.0):
                logger.warning("Trade skipped due to slippage.")
                send_message(token, chat_id, "Trade skipped due to slippage.")
                return {}
    except Exception as err:  # pragma: no cover - network
        logger.warning("Slippage check failed: %s", err)

    if (
        config.get("liquidity_check", True)
        and hasattr(exchange, "fetch_order_book")
        and not has_liquidity(amount)
    ):
        send_message(token, chat_id, "Insufficient liquidity for order size")
        return {}

    orders: List[Dict] = []
    if config.get("twap_enabled", False) and config.get("twap_slices", 1) > 1:
        slices = config.get("twap_slices", 1)
        delay = config.get("twap_interval_seconds", 1)
        slice_amount = amount / slices
        for i in range(slices):
            if (
                config.get("liquidity_check", True)
                and hasattr(exchange, "fetch_order_book")
                and not has_liquidity(slice_amount)
            ):
                send_message(
                    token, chat_id, "Insufficient liquidity during TWAP execution"
                )
                break
            order = place(slice_amount)
            if order:
                log_trade(order)
                if (config or {}).get("tax_tracking", {}).get("enabled"):
                    try:
                        if order.get("side") == "buy":
                            tax_logger.record_entry(order)
                        else:
                            tax_logger.record_exit(order)
                    except Exception:
                        pass
                orders.append(order)
                send_message(
                    token, chat_id, f"TWAP slice {i+1}/{slices} executed: {order}"
                )
                oid = (
                    order.get("id")
                    or order.get("order_id")
                    or order.get("tx_hash")
                    or order.get("txid")
                )
                logger.info(
                    "TWAP slice %s/%s %s %s %.8f executed (id/tx: %s)",
                    i + 1,
                    slices,
                    side,
                    symbol,
                    slice_amount,
                    oid,
                )
            if i < slices - 1:
                time.sleep(delay)
    else:
        order = place(amount)
        if order:
            log_trade(order)
            if (config or {}).get("tax_tracking", {}).get("enabled"):
                try:
                    if order.get("side") == "buy":
                        tax_logger.record_entry(order)
                    else:
                        tax_logger.record_exit(order)
                except Exception:
                    pass
            orders.append(order)
            send_message(token, chat_id, f"Order executed: {order}")
            oid = (
                order.get("id")
                or order.get("order_id")
                or order.get("tx_hash")
                or order.get("txid")
            )
            logger.info(
                "Order executed %s %s %.8f (id/tx: %s)",
                side,
                symbol,
                amount,
                oid,
            )

    if len(orders) == 1:
        return orders[0]
    return {"orders": orders}


async def execute_trade_async(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str,
    chat_id: str,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
) -> Dict:
    """Asynchronous version of :func:`execute_trade`. It supports both
    ``ccxt.pro`` exchanges and the threaded ``KrakenWSClient`` fallback."""

    msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
    else:
        try:
            if use_websocket and ws_client is not None and not ccxtpro:
                order = ws_client.add_order(symbol, side, amount)
            elif asyncio.iscoroutinefunction(
                getattr(exchange, "create_market_order", None)
            ):
                order = await exchange.create_market_order(symbol, side, amount)
            else:
                order = await asyncio.to_thread(
                    exchange.create_market_order, symbol, side, amount
                )
        except Exception as e:  # pragma: no cover - network
            send_message(token, chat_id, f"Order failed: {e}")
            return {}
    send_message(token, chat_id, f"Order executed: {order}")
    oid = (
        order.get("id")
        or order.get("order_id")
        or order.get("tx_hash")
        or order.get("txid")
    )
    logger.info(
        "Order executed %s %s %.8f (id/tx: %s)",
        side,
        symbol,
        amount,
        oid,
    )
    log_trade(order)
    logger.info(
        "Order executed - id=%s side=%s amount=%s price=%s dry_run=%s",
        order.get("id"),
        order.get("side"),
        order.get("amount"),
        order.get("price") or order.get("average"),
        dry_run,
    )
    if (config or {}).get("tax_tracking", {}).get("enabled"):
        try:
            if order.get("side") == "buy":
                tax_logger.record_entry(order)
            else:
                tax_logger.record_exit(order)
        except Exception:
            pass
    return order


def place_stop_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    token: str,
    chat_id: str,
    dry_run: bool = True,
) -> Dict:
    """Submit a stop-loss order on the exchange."""
    msg = f"Placing stop {side} order for {amount} {symbol} at {stop_price:.2f}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "stop": stop_price,
            "dry_run": True,
        }
    else:
        try:
            order = exchange.create_order(
                symbol,
                "stop_market",
                side,
                amount,
                params={"stopPrice": stop_price},
            )
        except Exception as e:
            send_message(token, chat_id, f"Stop order failed: {e}")
            return {}
    send_message(token, chat_id, f"Stop order submitted: {order}")
    log_trade(order, is_stop=True)
    return order

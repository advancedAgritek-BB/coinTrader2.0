from __future__ import annotations

import os
import time
try:
    import ccxt  # type: ignore
    from ccxt.base.errors import NetworkError, RateLimitExceeded, ExchangeError
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
    NetworkError = RateLimitExceeded = ExchangeError = Exception
import asyncio
from typing import Dict, Optional, Tuple, List

from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot import tax_logger
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.env import env_or_prompt
from crypto_bot.utils import kraken
from .kraken_client import KrakenClient


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client.

    When ``use_websocket`` is enabled a :class:`KrakenWSClient` is returned for
    realtime trading on Kraken. Only the standard ``ccxt`` library is used for
    exchange access.
    """

    raw_ex = config.get("exchange", "coinbase")
    if isinstance(raw_ex, dict):
        exchange_cfg = dict(raw_ex)
        exchange_name = exchange_cfg.get("name", "coinbase")
    else:
        exchange_cfg = {}
        exchange_name = raw_ex
    use_ws = config.get("use_websocket", False)
    requested_private_ws = config.get("kraken", {}).get(
        "use_private_ws", kraken.use_private_ws
    )
    use_private_ws = requested_private_ws and kraken.use_private_ws

    ws_client: Optional[KrakenWSClient] = None
    api_key = env_or_prompt("API_KEY", "Enter API key: ") or None
    api_secret = env_or_prompt("API_SECRET", "Enter API secret: ") or None

    params = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    }
    timeout = exchange_cfg.get("request_timeout_ms")
    if timeout:
        params["timeout"] = int(timeout)

    if exchange_cfg.get("max_concurrency") is not None:
        max_conc = int(exchange_cfg["max_concurrency"])
    else:
        max_conc = None

    if exchange_name == "coinbase":
        params["password"] = os.getenv("API_PASSPHRASE")
        exchange = ccxt.coinbase(params)
    elif exchange_name == "kraken":
        exchange = kraken.get_client(api_key, api_secret)

        if use_ws:
            if use_private_ws and not (api_key and api_secret):
                use_private_ws = kraken.use_private_ws = False
            ws_token = os.getenv("KRAKEN_WS_TOKEN") if use_private_ws else None
            if use_private_ws and not ws_token and api_key and api_secret:
                try:
                    ws_token = kraken.get_ws_token(api_key, api_secret)
                except Exception as err:
                    logger.warning("Failed to get WS token: %s", err)
                    use_private_ws = kraken.use_private_ws = False
            if use_private_ws or not requested_private_ws:
                try:
                    ws_client = KrakenWSClient(
                        api_key,
                        api_secret,
                        ws_token=ws_token if use_private_ws else None,
                        exchange=exchange,
                    )
                except Exception as err:  # pragma: no cover - optional dependency
                    logger.warning("Failed to initialize Kraken WS client: %s", err)
                    ws_client = None

        exchange = ccxt.kraken(params)
        exchange = KrakenClient(
            ccxt.kraken(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                }
            )
        )
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    exchange.options["ws_scan"] = config.get("use_websocket", False)
    if max_conc is not None:
        setattr(exchange, "max_concurrency", max_conc)

    return exchange, ws_client


def get_exchanges(config) -> Dict[str, Tuple[ccxt.Exchange, Optional[KrakenWSClient]]]:
    """Return exchange instances for all configured CEXes."""
    names = config.get("exchanges")
    if not names:
        raw_ex = config.get("exchange")
        if isinstance(raw_ex, dict):
            ex_name = raw_ex.get("name")
        else:
            ex_name = raw_ex
        primary = config.get("primary_exchange") or ex_name
        names = [name for name in [primary, config.get("secondary_exchange")] if name]
    result: Dict[str, Tuple[ccxt.Exchange, Optional[KrakenWSClient]]] = {}
    for name in names:
        cfg = dict(config)
        base_ex = config.get("exchange")
        if isinstance(base_ex, dict):
            new_ex = dict(base_ex)
            new_ex["name"] = name
            cfg["exchange"] = new_ex
        else:
            cfg["exchange"] = name
        result[name] = get_exchange(cfg)
    return result


def _resolve_notifier(
    token: str | TelegramNotifier | None,
    chat_id: Optional[str],
    notifier: Optional[TelegramNotifier],
) -> TelegramNotifier:
    """Return a ready-to-use :class:`TelegramNotifier` instance.

    The ``token`` argument may be a bot token string, an existing
    :class:`TelegramNotifier`, or ``None`` when ``notifier`` is provided.
    """

    if notifier is not None:
        return notifier
    if isinstance(token, TelegramNotifier):
        return token
    if token is None or chat_id is None:
        raise ValueError("token/chat_id or notifier must be provided")
    return TelegramNotifier(token, chat_id)


def _has_liquidity(order_book: Dict, side: str, order_size: float) -> bool:
    book = order_book.get("asks" if side == "buy" else "bids", [])
    vol = 0.0
    for _, qty in book:
        vol += qty
        if vol >= order_size:
            return True
    return False


async def _has_liquidity_async(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    order_size: float,
    config: Dict,
    notifier: TelegramNotifier,
) -> bool:
    try:
        depth = config.get("liquidity_depth", 10)
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_order_book", None)):
            ob = await exchange.fetch_order_book(symbol, limit=depth)
        else:
            ob = await asyncio.to_thread(exchange.fetch_order_book, symbol, depth)
        book = ob["asks" if side == "buy" else "bids"]
        vol = 0.0
        for _, qty in book:
            vol += qty
            if vol >= order_size:
                return True
        return False
    except Exception as err:
        err_msg = notifier.notify(f"Order book error: {err}")
        if err_msg:
            logger.error("Failed to send message: %s", err_msg)
        return False


def _evaluate_slippage(
    slippage: float,
    order_book: Optional[Dict],
    side: str,
    amount: float,
    config: Dict,
    notifier: TelegramNotifier,
) -> Tuple[bool, bool]:
    """Common slippage evaluation logic.

    Returns a tuple ``(force_twap, skip_trade)``. ``order_book`` may be
    ``None`` when only slippage should be considered (async path).
    """

    force_twap = False

    if slippage > config.get("max_slippage_pct", 1.0):
        if config.get("twap_enabled", False):
            force_twap = True
        else:
            logger.warning("Trade skipped due to slippage.")
            err_msg = notifier.notify("Trade skipped due to slippage.")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return force_twap, True

    if order_book:
        book = order_book["asks" if side == "buy" else "bids"]
        total_vol = sum(qty for _, qty in book)
        max_use = total_vol * config.get("max_liquidity_usage", 0.8)
        if amount > max_use:
            logger.warning("Trade skipped due to low liquidity: %s > %s", amount, max_use)
            err_msg = notifier.notify("Insufficient liquidity for order size")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            return force_twap, True

    return force_twap, False


def _check_slippage_sync(
    order_book: Dict,
    side: str,
    amount: float,
    config: Dict,
    notifier: TelegramNotifier,
) -> Tuple[bool, bool]:
    """Return ``(force_twap, skip_trade)`` based on slippage and liquidity."""

    if not order_book:
        return False, False

    slippage = estimate_book_slippage(order_book, side, amount)
    return _evaluate_slippage(slippage, order_book, side, amount, config, notifier)


async def _check_slippage_async(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    config: Dict,
    notifier: TelegramNotifier,
) -> bool:
    try:
        depth = config.get("liquidity_depth", 10)
        slippage = await estimate_book_slippage_async(exchange, symbol, side, amount, depth)
        _, skip = _evaluate_slippage(slippage, None, side, amount, config, notifier)
        return skip
    except Exception as err:  # pragma: no cover - network
        logger.warning("Slippage check failed: %s", err)
        return False


async def _place_order_common(
    symbol: str,
    side: str,
    size: float,
    notifier: TelegramNotifier,
    dry_run: bool,
    max_retries: int,
    poll_timeout: int,
    score: float,
    config: Dict,
    *,
    create_market,
    fetch_order,
    sleep,
    create_limit=None,
    fetch_ticker=None,
) -> List[Dict]:
    """Shared implementation for order placement."""

    if dry_run:
        return [{"symbol": symbol, "side": side, "amount": size, "dry_run": True}]

    remaining = size
    orders: List[Dict] = []

    while remaining > 0:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                if score > 0.8 and create_limit is not None:
                    price = None
                    if fetch_ticker is not None:
                        try:
                            t = await fetch_ticker()
                            bid = t.get("bid")
                            ask = t.get("ask")
                            if bid and ask:
                                price = (bid + ask) / 2
                        except Exception as err:
                            logger.warning("Limit price fetch failed: %s", err)
                    if price:
                        params = {"postOnly": True}
                        if config.get("hidden_limit"):
                            params["hidden"] = True
                        order = await create_limit(symbol, side, remaining, price, params)
                        break

                order = await create_market(symbol, side, remaining)
                break
            except (NetworkError, RateLimitExceeded) as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Retry %s placing %s %s due to %s",
                        attempt + 1,
                        side,
                        symbol,
                        exc,
                    )
                    await sleep(delay)
                    delay *= 2
                    continue
                raise
            except ExchangeError:
                raise
            except Exception as exc:
                logger.exception("Order placement failed: %s", exc)
                err_msg = notifier.notify(f"Order failed: {exc}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                raise

        start = time.time()
        filled = 0.0
        while time.time() - start < poll_timeout:
            try:
                info = await fetch_order(order["id"], symbol)
                filled = float(info.get("filled") or 0.0)
                order.update(info)
                if 0 < filled < remaining:
                    err_pf = notifier.notify(f"Partial fill: {filled}/{remaining} {symbol}")
                    if err_pf:
                        logger.error("Failed to send message: %s", err_pf)
                if info.get("status") == "closed":
                    break
            except Exception as err:
                logger.debug("Order polling error: %s", err)
            await sleep(1)

        orders.append(order)
        if filled and filled < remaining:
            remaining -= filled
        else:
            remaining = 0
    return orders


def _place_order_sync(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    size: float,
    notifier: TelegramNotifier,
    dry_run: bool,
    max_retries: int,
    poll_timeout: int,
    score: float,
    config: Dict,
) -> List[Dict]:
    """Place ``size`` amount and wait for completion (synchronous wrapper)."""

    async def create_market(sym: str, s: str, amt: float):
        if ws_client is not None:
            return ws_client.add_order(sym, s, amt)
        return await asyncio.to_thread(exchange.create_market_order, sym, s, amt)

    async def fetch_order(order_id: str, sym: str):
        return await asyncio.to_thread(exchange.fetch_order, order_id, sym)

    async def sleep(delay: float):
        await asyncio.to_thread(time.sleep, delay)

    create_limit = None
    fetch_ticker = None
    if hasattr(exchange, "create_limit_order"):
        async def create_limit(sym: str, s: str, amt: float, price: float, params: Dict):
            return await asyncio.to_thread(
                exchange.create_limit_order, sym, s, amt, price, params
            )

        if hasattr(exchange, "fetch_ticker"):
            async def fetch_ticker():
                return await asyncio.to_thread(exchange.fetch_ticker, symbol)

    return asyncio.run(
        _place_order_common(
            symbol,
            side,
            size,
            notifier,
            dry_run,
            max_retries,
            poll_timeout,
            score,
            config,
            create_market=create_market,
            fetch_order=fetch_order,
            sleep=sleep,
            create_limit=create_limit,
            fetch_ticker=fetch_ticker,
        )
    )


async def _place_order_async(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    size: float,
    notifier: TelegramNotifier,
    dry_run: bool,
    max_retries: int,
    poll_timeout: int,
    score: float,
    config: Dict,
    use_websocket: bool,
) -> List[Dict]:
    """Async variant of order placement."""

    async def create_market(sym: str, s: str, amt: float):
        if use_websocket and ws_client is not None:
            return ws_client.add_order(sym, s, amt)
        if asyncio.iscoroutinefunction(getattr(exchange, "create_market_order", None)):
            return await exchange.create_market_order(sym, s, amt)
        return await asyncio.to_thread(exchange.create_market_order, sym, s, amt)

    async def fetch_order(order_id: str, sym: str):
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_order", None)):
            return await exchange.fetch_order(order_id, sym)
        return await asyncio.to_thread(exchange.fetch_order, order_id, sym)

    async def sleep(delay: float):
        await asyncio.sleep(delay)

    create_limit = None
    fetch_ticker = None
    if hasattr(exchange, "create_limit_order"):
        async def create_limit(sym: str, s: str, amt: float, price: float, params: Dict):
            if asyncio.iscoroutinefunction(getattr(exchange, "create_limit_order", None)):
                return await exchange.create_limit_order(sym, s, amt, price, params)
            return await asyncio.to_thread(
                exchange.create_limit_order, sym, s, amt, price, params
            )

        if hasattr(exchange, "fetch_ticker"):
            async def fetch_ticker():
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                    return await exchange.fetch_ticker(symbol)
                return await asyncio.to_thread(exchange.fetch_ticker, symbol)

    return await _place_order_common(
        symbol,
        side,
        size,
        notifier,
        dry_run,
        max_retries,
        poll_timeout,
        score,
        config,
        create_market=create_market,
        fetch_order=fetch_order,
        sleep=sleep,
        create_limit=create_limit,
        fetch_ticker=fetch_ticker,
    )




def execute_trade(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str | TelegramNotifier | None = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
    score: float = 0.0,
    max_retries: int = 3,
    poll_timeout: int = 60,
) -> Dict:
    """Place a market or limit order with optional retries.

    Parameters
    ----------
    max_retries:
        Number of attempts when API calls fail with transient errors.
        Defaults to ``3``.
    poll_timeout:
        Seconds to wait for each order to close before placing the
        remaining quantity. Defaults to ``60``.
    """
    notifier = _resolve_notifier(token, chat_id, notifier)
    if use_websocket and ws_client is None and not dry_run:
        raise ValueError("WebSocket trading enabled but ws_client is missing")
    config = config or {}

    depth = config.get("liquidity_depth", 10)
    order_book: Dict = {}
    if hasattr(exchange, "fetch_order_book"):
        try:
            order_book = exchange.fetch_order_book(symbol, limit=depth)
        except Exception as err:  # pragma: no cover - network
            logger.warning("Order book fetch failed: %s", err)
            err = notifier.notify(f"Order book error: {err}")
            if err:
                logger.error("Failed to send message: %s", err)

    def has_liquidity(order_size: float) -> bool:
        return _has_liquidity(order_book, side, order_size)

    def place(size: float) -> List[Dict]:
        return _place_order_sync(
            exchange,
            ws_client,
            symbol,
            side,
            size,
            notifier,
            dry_run,
            max_retries,
            poll_timeout,
            score,
            config,
        )

    err = notifier.notify(f"Placing {side} order for {amount} {symbol}")
    if err:
        logger.error("Failed to send message: %s", err)
    force_twap, skip_trade = _check_slippage_sync(
        order_book, side, amount, config, notifier
    )
    if skip_trade:
        return {}

    if config.get("liquidity_check", True) and order_book and not has_liquidity(amount):
        notifier.notify("Insufficient liquidity for order size")
        return {}

    orders: List[Dict] = []
    if (force_twap or config.get("twap_enabled", False)) and config.get("twap_slices", 1) > 1:
        slices = config.get("twap_slices", 1)
        delay = config.get("twap_interval_seconds", 1)
        slice_amount = amount / slices
        for i in range(slices):
            if config.get("liquidity_check", True) and order_book and not has_liquidity(slice_amount):
                err_liq = notifier.notify(
                    "Insufficient liquidity during TWAP execution"
                )
                if err_liq:
                    logger.error("Failed to send message: %s", err_liq)
                break
            placed = place(slice_amount)
            for order in placed:
                if dry_run:
                    try:
                        t = exchange.fetch_ticker(symbol)
                        order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                    except Exception:
                        order["price"] = (config or {}).get("entry_price", 0.0)
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
                err_slice = notifier.notify(
                    f"TWAP slice {i+1}/{slices} executed: {order}"
                )
                if err_slice:
                    logger.error("Failed to send message: %s", err_slice)
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
        placed = place(amount)
        for order in placed:
            if dry_run:
                try:
                    t = exchange.fetch_ticker(symbol)
                    order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                except Exception:
                    order["price"] = (config or {}).get("entry_price", 0.0)
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
            err_exec = notifier.notify(f"Order executed: {order}")
            if err_exec:
                logger.error("Failed to send message: %s", err_exec)
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

    if not dry_run:
        try:
            sync_positions(exchange)
        except Exception as exc:  # pragma: no cover - optional
            logger.error("Position sync failed: %s", exc)

    if len(orders) == 1:
        return orders[0]
    return {"orders": orders}


async def execute_trade_async(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str | TelegramNotifier | None = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
    score: float = 0.0,
    max_retries: int = 3,
    poll_timeout: int = 60,
) -> Dict:
    """Asynchronous version of :func:`execute_trade` with retry support.

    Simplified async trade execution used in tests.
    """

    if notifier is None:
        if isinstance(token, TelegramNotifier):
            notifier = token
        else:
            if token is None or chat_id is None:
                raise ValueError("token/chat_id or notifier must be provided")
            notifier = TelegramNotifier(token, chat_id)

    if use_websocket and ws_client is None and not dry_run:
        raise ValueError("WebSocket trading enabled but ws_client is missing")

    msg = f"Placing {side} order for {amount} {symbol}"
    err = notifier.notify(msg)
    if err:
        logger.error("Failed to send message: %s", err)

    config = config or {}

    skip = await _check_slippage_async(exchange, symbol, side, amount, config, notifier)
    if skip:
        return {}

    async def has_liquidity(order_size: float) -> bool:
        return await _has_liquidity_async(
            exchange, symbol, side, order_size, config, notifier
        )

    if (
        config.get("liquidity_check", True)
        and hasattr(exchange, "fetch_order_book")
        and not await has_liquidity(amount)
    ):
        notifier.notify("Insufficient liquidity for order size")
        return {}

    async def place(size: float) -> List[Dict]:
        return await _place_order_async(
            exchange,
            ws_client,
            symbol,
            side,
            size,
            notifier,
            dry_run,
            max_retries,
            poll_timeout,
            score,
            config,
            use_websocket,
        )

    all_orders: List[Dict] = []

    if config and config.get("twap_enabled", False) and config.get("twap_slices", 1) > 1:
        slices = config.get("twap_slices", 1)
        delay = config.get("twap_interval_seconds", 1)
        slice_amount = amount / slices
        for i in range(slices):
            if (
                config.get("liquidity_check", True)
                and hasattr(exchange, "fetch_order_book")
                and not await has_liquidity(slice_amount)
            ):
                err_liq = notifier.notify("Insufficient liquidity during TWAP execution")
                if err_liq:
                    logger.error("Failed to send message: %s", err_liq)
                break
            placed = await place(slice_amount)
            for order in placed:
                if dry_run:
                    try:
                        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                            t = await exchange.fetch_ticker(symbol)
                        else:
                            t = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                        order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                    except Exception:
                        order["price"] = (config or {}).get("entry_price", 0.0)
                log_trade(order)
                if (config or {}).get("tax_tracking", {}).get("enabled"):
                    try:
                        if order.get("side") == "buy":
                            tax_logger.record_entry(order)
                        else:
                            tax_logger.record_exit(order)
                    except Exception:
                        pass
                all_orders.append(order)
                err_slice = notifier.notify(f"TWAP slice {i+1}/{slices} executed: {order}")
                if err_slice:
                    logger.error("Failed to send message: %s", err_slice)
            if i < slices - 1:
                await asyncio.sleep(delay)
    else:
        placed = await place(amount)
        for order in placed:
            if dry_run:
                try:
                    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                        t = await exchange.fetch_ticker(symbol)
                    else:
                        t = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                    order["price"] = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                except Exception:
                    order["price"] = (config or {}).get("entry_price", 0.0)
            log_trade(order)
            if (config or {}).get("tax_tracking", {}).get("enabled"):
                try:
                    if order.get("side") == "buy":
                        tax_logger.record_entry(order)
                    else:
                        tax_logger.record_exit(order)
                except Exception:
                    pass
            all_orders.append(order)
            err_exec = notifier.notify(f"Order executed: {order}")
            if err_exec:
                logger.error("Failed to send message: %s", err_exec)

    if not dry_run:
        try:
            await sync_positions_async(exchange)
        except Exception as exc:  # pragma: no cover - optional
            logger.error("Position sync failed: %s", exc)

    if len(all_orders) == 1:
        return all_orders[0]
    return {"orders": all_orders}


def place_stop_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    token: str | TelegramNotifier | None = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    dry_run: bool = True,
    max_retries: int = 3,
) -> Dict:
    """Submit a stop-loss order on the exchange.

    Parameters
    ----------
    max_retries:
        Retry attempts when order placement fails. Defaults to ``3``.
    """
    notifier = _resolve_notifier(token, chat_id, notifier)

    msg = f"Placing stop {side} order for {amount} {symbol} at {stop_price:.2f}"
    err = notifier.notify(msg)
    if err:
        logger.error("Failed to send message: %s", err)
    if dry_run:
        order = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "stop": stop_price,
            "dry_run": True,
        }
    else:
        delay = 1.0
        order = None
        for attempt in range(max_retries):
            try:
                order = exchange.create_order(
                    symbol,
                    "stop_market",
                    side,
                    amount,
                    params={"stopPrice": stop_price},
                )
                break
            except (NetworkError, RateLimitExceeded) as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Retry %s placing stop %s %s due to %s",
                        attempt + 1,
                        side,
                        symbol,
                        exc,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except ExchangeError:
                raise
            except Exception as e:
                logger.exception("Stop order failed: %s", e)
                err_msg = notifier.notify(f"Stop order failed: {e}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {}
        if order is None:
            return {}
    err = notifier.notify(f"Stop order submitted: {order}")
    if err:
        logger.error("Failed to send message: %s", err)
    log_trade(order, is_stop=True)
    return order


def estimate_book_slippage(order_book: Dict[str, List[List[float]]], side: str, amount: float) -> float:
    """Estimate slippage percentage for a market order using an order book."""

    if amount <= 0:
        return 0.0

    levels = order_book.get("asks" if side == "buy" else "bids") or []
    if not levels:
        return 0.0

    best_price = float(levels[0][0])
    remaining = amount
    cost = 0.0
    filled = 0.0
    for values in levels:
        if len(values) < 2:
            logger.warning("Invalid slippage values: %s", values)
            return 0.0
        price, qty = values[:2]
        take = min(remaining, float(qty))
        cost += take * float(price)
        filled += take
        remaining -= take
        if remaining <= 0:
            break

    if filled == 0:
        return 0.0
    avg_price = cost / filled
    if side == "buy":
        return (avg_price - best_price) / best_price
    else:
        return (best_price - avg_price) / best_price


async def estimate_book_slippage_async(
    exchange,
    symbol: str,
    side: str,
    amount: float,
    depth: int = 10,
) -> float:
    """Fetch order book and return slippage estimate asynchronously."""

    fetch_fn = getattr(exchange, "fetch_order_book", None)
    if fetch_fn is None:
        return 0.0

    try:
        if asyncio.iscoroutinefunction(fetch_fn):
            book = await fetch_fn(symbol, limit=depth)
        else:
            book = await asyncio.to_thread(fetch_fn, symbol, depth)
    except Exception:  # pragma: no cover - network
        return 0.0

    return estimate_book_slippage(book, side, amount)


def sync_positions(exchange: ccxt.Exchange) -> List[Dict]:
    """Return open positions or open orders from ``exchange``."""

    fetch_positions = getattr(exchange, "fetch_positions", None)
    fetch_orders = getattr(exchange, "fetch_open_orders", None)

    result: List[Dict] = []
    try:
        if fetch_positions is not None:
            data = fetch_positions()
        elif fetch_orders is not None:
            data = fetch_orders()
        else:
            logger.error("Exchange missing position fetch methods")
            return []

        for pos in data:
            result.append(
                {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side") or pos.get("direction"),
                    "size": float(
                        pos.get("contracts")
                        or pos.get("amount")
                        or pos.get("size")
                        or 0.0
                    ),
                    "price": float(
                        pos.get("entryPrice")
                        or pos.get("average")
                        or pos.get("price")
                        or 0.0
                    ),
                }
            )
    except Exception as exc:  # pragma: no cover - optional
        logger.error("Failed to sync positions: %s", exc)
        return []

    return result


async def sync_positions_async(exchange: ccxt.Exchange) -> List[Dict]:
    """Asynchronous variant of :func:`sync_positions`."""

    fetch_positions = getattr(exchange, "fetch_positions", None)
    fetch_orders = getattr(exchange, "fetch_open_orders", None)

    try:
        if fetch_positions is not None:
            if asyncio.iscoroutinefunction(fetch_positions):
                data = await fetch_positions()
            else:
                data = await asyncio.to_thread(fetch_positions)
        elif fetch_orders is not None:
            if asyncio.iscoroutinefunction(fetch_orders):
                data = await fetch_orders()
            else:
                data = await asyncio.to_thread(fetch_orders)
        else:
            logger.error("Exchange missing position fetch methods")
            return []
    except Exception as exc:  # pragma: no cover - optional
        logger.error("Failed to sync positions: %s", exc)
        return []

    result: List[Dict] = []
    for pos in data:
        result.append(
            {
                "symbol": pos.get("symbol"),
                "side": pos.get("side") or pos.get("direction"),
                "size": float(
                    pos.get("contracts")
                    or pos.get("amount")
                    or pos.get("size")
                    or 0.0
                ),
                "price": float(
                    pos.get("entryPrice")
                    or pos.get("average")
                    or pos.get("price")
                    or 0.0
                ),
            }
        )

    return result

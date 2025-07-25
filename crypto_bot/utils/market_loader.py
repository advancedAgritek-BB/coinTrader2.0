"""Utilities for loading trading symbols and fetching OHLCV data."""

from typing import Iterable, List, Dict, Any, Deque
from dataclasses import dataclass
import asyncio
import inspect
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import ccxt
import aiohttp
import datetime
import base58
from .gecko import gecko_request
import contextlib

from .token_registry import (
    TOKEN_MINTS,
    get_mint_from_gecko,
    fetch_from_helius,
)

from .telegram import TelegramNotifier
from .logger import LOG_DIR, setup_logger
from .constants import NON_SOLANA_BASES


_last_snapshot_time = 0

logger = setup_logger(__name__, LOG_DIR / "bot.log")

failed_symbols: Dict[str, Dict[str, Any]] = {}
# Track WebSocket OHLCV failures per symbol
WS_FAIL_COUNTS: Dict[str, int] = {}
RETRY_DELAY = 300
MAX_RETRY_DELAY = 3600
# Default timeout when fetching OHLCV data
OHLCV_TIMEOUT = 60
# Default timeout when fetching OHLCV data over WebSocket
WS_OHLCV_TIMEOUT = 60
# REST requests occasionally face Cloudflare delays up to a minute
REST_OHLCV_TIMEOUT = 90
# Number of consecutive failures allowed before disabling a symbol
MAX_OHLCV_FAILURES = 10
MAX_WS_LIMIT = 500
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
STATUS_UPDATES = True
SEMA: asyncio.Semaphore | None = None

# Mapping of common symbols to CoinGecko IDs for OHLC fallback
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# Cache GeckoTerminal pool addresses and metadata per symbol
# Mapping: symbol -> (pool_addr, volume, reserve, price, limit)
GECKO_POOL_CACHE: dict[str, tuple[str, float, float, float, int]] = {}
# Cache Kraken listing timestamps per symbol
KRAKEN_LISTING_CACHE: dict[str, int | None] = {}
GECKO_SEMAPHORE = asyncio.Semaphore(10)
# Track symbols that don't exist on GeckoTerminal to avoid repeated lookups
GECKO_UNAVAILABLE: set[str] = set()

# Batch queues for OHLCV updates keyed by request parameters
_OHLCV_BATCH_QUEUES: Dict[tuple, asyncio.Queue] = {}
_OHLCV_BATCH_TASKS: Dict[tuple, asyncio.Task] = {}


@dataclass
class _OhlcvBatchRequest:
    exchange: Any
    cache: Dict[str, pd.DataFrame]
    symbols: List[str]
    timeframe: str
    limit: int
    start_since: int | None
    use_websocket: bool
    force_websocket_history: bool
    config: Dict
    max_concurrent: int | None
    notifier: TelegramNotifier | None
    future: asyncio.Future


async def _ohlcv_batch_worker(
    key: tuple,
    queue: asyncio.Queue,
    batch_size: int,
    delay: float,
) -> None:
    """Process queued OHLCV requests."""
    if not isinstance(batch_size, int) or batch_size < 1:
        logger.warning(
            "Invalid batch_size %r passed to _ohlcv_batch_worker; using 1",
            batch_size,
        )
        batch_size = 1
    try:
        while True:
            try:
                first: _OhlcvBatchRequest = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                if queue.empty():
                    break
                continue

            reqs = [first]
            start = time.monotonic()
            while len(reqs) < batch_size:
                timeout = delay - (time.monotonic() - start)
                if timeout <= 0:
                    break
                try:
                    reqs.append(await asyncio.wait_for(queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            union_symbols: List[str] = []
            for r in reqs:
                union_symbols.extend(r.symbols)
            # Deduplicate while preserving order
            seen = set()
            union_symbols = [s for s in union_symbols if not (s in seen or seen.add(s))]

            base = reqs[0]
            cache = await _update_ohlcv_cache_inner(
                base.exchange,
                base.cache,
                union_symbols,
                timeframe=base.timeframe,
                limit=base.limit,
                start_since=base.start_since,
                use_websocket=base.use_websocket,
                force_websocket_history=base.force_websocket_history,
                config=base.config,
                max_concurrent=base.max_concurrent,
                notifier=base.notifier,
            )

            for r in reqs:
                if r.cache is not cache:
                    for s in r.symbols:
                        if s in cache:
                            r.cache[s] = cache[s]
                if not r.future.done():
                    r.future.set_result(r.cache)
                queue.task_done()
    finally:
        _OHLCV_BATCH_TASKS.pop(key, None)

# Valid characters for Solana addresses
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Quote currencies eligible for Coinbase fallback
SUPPORTED_USD_QUOTES = {"USD", "USDC", "USDT"}

def _is_valid_base_token(token: str) -> bool:
    """Return ``True`` if ``token`` is known or looks like a Solana mint."""
    token_upper = token.upper()
    if token_upper in NON_SOLANA_BASES:
        return False
    if token_upper in TOKEN_MINTS:
        return True
    if not isinstance(token, str):
        return False
    if not (32 <= len(token) <= 44):
        return False
    try:
        return len(base58.b58decode(token)) == 32 and all(
            c in BASE58_ALPHABET for c in token
        )
    except Exception:
        return False


def configure(
    ohlcv_timeout: int | float | None = None,
    max_failures: int | None = None,
    max_ws_limit: int | None = None,
    status_updates: bool | None = None,
    ws_ohlcv_timeout: int | float | None = None,
    rest_ohlcv_timeout: int | float | None = None,
    max_concurrent: int | None = None,
    gecko_limit: int | None = None,
) -> None:
    """Configure module-wide settings."""
    global OHLCV_TIMEOUT, MAX_OHLCV_FAILURES, MAX_WS_LIMIT, STATUS_UPDATES, SEMA, GECKO_SEMAPHORE
    cfg = None
    if ohlcv_timeout is None or max_failures is None:
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    if ohlcv_timeout is None and cfg is not None:
        cfg_val = cfg.get("ohlcv_timeout")
        if cfg_val is not None:
            ohlcv_timeout = cfg_val
    if max_failures is None and cfg is not None:
        cfg_val = cfg.get("max_ohlcv_failures")
        if cfg_val is not None:
            max_failures = cfg_val
    if ohlcv_timeout is not None:
        try:
            val = max(1, int(ohlcv_timeout))
            OHLCV_TIMEOUT = val
            WS_OHLCV_TIMEOUT = val
            REST_OHLCV_TIMEOUT = val
        except (TypeError, ValueError):
            logger.warning(
                "Invalid ohlcv_timeout %s; using default %s",
                ohlcv_timeout,
                OHLCV_TIMEOUT,
            )
    if ws_ohlcv_timeout is not None:
        try:
            WS_OHLCV_TIMEOUT = max(1, int(ws_ohlcv_timeout))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid WS_OHLCV_TIMEOUT %s; using default %s",
                ws_ohlcv_timeout,
                WS_OHLCV_TIMEOUT,
            )
    if rest_ohlcv_timeout is not None:
        try:
            REST_OHLCV_TIMEOUT = max(1, int(rest_ohlcv_timeout))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid REST_OHLCV_TIMEOUT %s; using default %s",
                rest_ohlcv_timeout,
                REST_OHLCV_TIMEOUT,
            )
    if max_failures is not None:
        try:
            MAX_OHLCV_FAILURES = max(1, int(max_failures))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MAX_OHLCV_FAILURES %s; using default %s",
                max_failures,
                MAX_OHLCV_FAILURES,
            )
    if max_ws_limit is None:
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_val = cfg.get("max_ws_limit")
            if cfg_val is not None:
                max_ws_limit = cfg_val
        except Exception:
            pass
    if max_ws_limit is not None:
        try:
            MAX_WS_LIMIT = max(1, int(max_ws_limit))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MAX_WS_LIMIT %s; using default %s",
                max_ws_limit,
                MAX_WS_LIMIT,
            )
    if status_updates is not None:
        STATUS_UPDATES = bool(status_updates)
    if max_concurrent is None:
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_val = cfg.get("max_concurrent_ohlcv")
            if cfg_val is not None:
                max_concurrent = cfg_val
        except Exception:
            pass
    if max_concurrent is not None:
        try:
            val = int(max_concurrent)
            if val < 1:
                raise ValueError
            SEMA = asyncio.Semaphore(val)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_concurrent %s; disabling semaphore", max_concurrent
            )
            SEMA = None

    if gecko_limit is not None:
        try:
            val = int(gecko_limit)
            if val < 1:
                raise ValueError
            GECKO_SEMAPHORE = asyncio.Semaphore(val)
        except (TypeError, ValueError):
            logger.warning("Invalid gecko_limit %s; using default", gecko_limit)


def is_symbol_type(pair_info: dict, allowed: List[str]) -> bool:
    """Return ``True`` if ``pair_info`` matches one of the ``allowed`` types.

    The heuristic checks common CCXT fields like ``type`` and boolean flags
    (``spot``, ``future``, ``swap``) along with nested ``info`` metadata.  If no
    explicit type can be determined, a pair is treated as ``spot`` by default.
    """

    allowed_set = {t.lower() for t in allowed}

    market_type = str(pair_info.get("type", "")).lower()
    if market_type:
        return market_type in allowed_set

    for key in ("spot", "future", "swap", "option"):
        if pair_info.get(key) and key in allowed_set:
            return True

    info = pair_info.get("info", {}) or {}
    asset_class = str(info.get("assetClass", "")).lower()
    if asset_class:
        if asset_class in allowed_set:
            return True
        if asset_class in ("perpetual", "swap") and "swap" in allowed_set:
            return True
        if asset_class in ("future", "futures") and "future" in allowed_set:
            return True

    contract_type = str(info.get("contractType", "")).lower()
    if contract_type:
        if contract_type in allowed_set:
            return True
        if "perp" in contract_type and "swap" in allowed_set:
            return True

    # default to spot if no derivative hints are present
    if "spot" in allowed_set:
        derivative_keys = (
            "future",
            "swap",
            "option",
            "expiry",
            "contract",
            "settlement",
        )
        if not any(k in pair_info for k in derivative_keys) and not any(
            k in info for k in derivative_keys
        ):
            return True

    return False


def timeframe_seconds(exchange, timeframe: str) -> int:
    """Return timeframe length in seconds."""
    if hasattr(exchange, "parse_timeframe"):
        try:
            return int(exchange.parse_timeframe(timeframe))
        except Exception:
            pass
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604800
    if unit == "M":
        return value * 2592000
    raise ValueError(f"Unknown timeframe {timeframe}")


async def _call_with_retry(func, *args, timeout=None, **kwargs):
    """Call ``func`` with fixed back-off on 520/522 errors."""

    attempts = 3
    delays = [5, 10, 20]
    for attempt in range(attempts):
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    asyncio.shield(func(*args, **kwargs)), timeout
                )
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except (ccxt.ExchangeError, ccxt.NetworkError) as exc:
            if (
                getattr(exc, "http_status", None) in (520, 522)
                and attempt < attempts - 1
            ):
                await asyncio.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise




async def load_kraken_symbols(
    exchange,
    exclude: Iterable[str] | None = None,
    config: Dict | None = None,
) -> List[str] | None:
    """Return a list of active trading pairs on Kraken.

    Parameters
    ----------
    exchange : ccxt Exchange
        Exchange instance connected to Kraken.
    exclude : Iterable[str] | None
        Symbols to exclude from the result.
    """

    exclude_set = set(exclude or [])
    if config and "exchange_market_types" in config:
        allowed_types = set(config["exchange_market_types"])
    else:
        allowed_types = set(getattr(exchange, "exchange_market_types", []))
        if not allowed_types:
            allowed_types = {"spot"}

    markets = None
    if getattr(exchange, "has", {}).get("fetchMarketsByType"):
        fetcher = getattr(exchange, "fetch_markets_by_type", None) or getattr(
            exchange, "fetchMarketsByType", None
        )
        if fetcher:
            markets = {}
            for m_type in allowed_types:
                try:
                    if asyncio.iscoroutinefunction(fetcher):
                        fetched = await fetcher(m_type)
                    else:
                        fetched = await asyncio.to_thread(fetcher, m_type)
                except TypeError:
                    params = {"type": m_type}
                    if asyncio.iscoroutinefunction(fetcher):
                        fetched = await fetcher(params)
                    else:
                        fetched = await asyncio.to_thread(fetcher, params)
                except Exception as exc:  # pragma: no cover - safety
                    logger.warning("fetch_markets_by_type failed: %s", exc)
                    continue
                if isinstance(fetched, dict):
                    for sym, info in fetched.items():
                        info.setdefault("type", m_type)
                        markets[sym] = info
                elif isinstance(fetched, list):
                    for info in fetched:
                        sym = info.get("symbol")
                        if sym:
                            info.setdefault("type", m_type)
                            markets[sym] = info
    if markets is None:
        if asyncio.iscoroutinefunction(getattr(exchange, "load_markets", None)):
            markets = await exchange.load_markets()
        else:
            markets = await asyncio.to_thread(exchange.load_markets)

    df = pd.DataFrame.from_dict(markets, orient="index")
    df.index.name = "symbol"
    if "symbol" in df.columns:
        df.drop(columns=["symbol"], inplace=True)
    df.reset_index(inplace=True)

    df["active"] = df.get("active", True).fillna(True)
    df["reason"] = None
    df.loc[~df["active"], "reason"] = "inactive"

    mask_type = df.apply(lambda r: is_symbol_type(r.to_dict(), allowed_types), axis=1)
    df.loc[df["reason"].isna() & ~mask_type, "reason"] = (
        "type mismatch ("
        + df.get("type", "unknown").fillna("unknown").astype(str)
        + ")"
    )

    df.loc[df["reason"].isna() & df["symbol"].isin(exclude_set), "reason"] = "excluded"

    symbols: List[str] = []
    for row in df.itertuples():
        if row.reason:
            logger.debug("Skipping symbol %s: %s", row.symbol, row.reason)
        else:
            logger.debug("Including symbol %s", row.symbol)
            symbols.append(row.symbol)

    if not symbols:
        logger.warning("No active trading pairs were discovered")
        return None

    return symbols


async def _fetch_ohlcv_async_inner(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> list | Exception:
    """Internal helper for :func:`fetch_ohlcv_async`."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchOHLCV"):
        ex_id = getattr(exchange, "id", "unknown")
        logger.warning("Exchange %s lacks fetchOHLCV capability", ex_id)
        return []
    if getattr(exchange, "timeframes", None) and timeframe not in getattr(
        exchange, "timeframes", {}
    ):
        ex_id = getattr(exchange, "id", "unknown")
        logger.warning("Timeframe %s not supported on %s", timeframe, ex_id)
        return []

    if timeframe in ("4h", "1d"):
        use_websocket = False


    try:
        if hasattr(exchange, "symbols"):
            if not exchange.symbols and hasattr(exchange, "load_markets"):
                try:
                    if asyncio.iscoroutinefunction(
                        getattr(exchange, "load_markets", None)
                    ):
                        await exchange.load_markets()
                    else:
                        await asyncio.to_thread(exchange.load_markets)
                except Exception as exc:
                    logger.warning("load_markets failed: %s", exc)
            if exchange.symbols and symbol not in exchange.symbols:
                logger.warning(
                    "Skipping unsupported symbol %s on %s",
                    symbol,
                    getattr(exchange, "id", "unknown"),
                )
                return []

        if not use_websocket and limit > 0:
            data_all: list = []
            while limit > 0:
                req_limit = min(limit, 720)
                params = {"symbol": symbol, "timeframe": timeframe, "limit": req_limit}
                if since is not None:
                    params["since"] = since
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    batch = await _call_with_retry(
                        exchange.fetch_ohlcv,
                        timeout=REST_OHLCV_TIMEOUT,
                        **params,
                    )
                else:
                    batch = await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **params,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                data_all.extend(batch)
                limit -= len(batch)
                if len(batch) < req_limit:
                    break
                since = batch[-1][0] + timeframe_seconds(exchange, timeframe) * 1000
            return data_all
        if (
            use_websocket
            and since is None
            and timeframe == "1m"
            and limit > MAX_WS_LIMIT
            and not force_websocket_history
        ):
            logger.info(
                "Skipping WebSocket OHLCV for %s limit %d exceeds %d",
                symbol,
                limit,
                MAX_WS_LIMIT,
            )
            use_websocket = False
            limit = min(limit, MAX_WS_LIMIT)
        if use_websocket and since is not None:
            try:
                seconds = timeframe_seconds(exchange, timeframe)
                candles_needed = int((time.time() - since) / seconds) + 1
                if candles_needed < limit:
                    limit = candles_needed
            except Exception:
                pass
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            params = inspect.signature(exchange.watch_ohlcv).parameters
            ws_limit = limit
            kwargs = {"symbol": symbol, "timeframe": timeframe, "limit": ws_limit}
            if since is not None and "since" in params:
                kwargs["since"] = since
                tf_sec = timeframe_seconds(exchange, timeframe)
                try:
                    if since > 1e10:
                        now_ms = int(time.time() * 1000)
                        expected = max(0, (now_ms - since) // (tf_sec * 1000))
                        ws_limit = max(1, min(ws_limit, int(expected) + 2))
                    else:
                        expected = max(0, (time.time() - since) // tf_sec)
                        ws_limit = max(1, min(ws_limit, int(expected) + 1))
                    kwargs["limit"] = ws_limit
                except Exception:
                    pass
            for attempt in range(3):
                try:
                    data = await _call_with_retry(
                        exchange.watch_ohlcv, timeout=WS_OHLCV_TIMEOUT, **kwargs
                    )
                    WS_FAIL_COUNTS[symbol] = 0
                    break
                except asyncio.CancelledError:
                    if hasattr(exchange, "close"):
                        if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                            with contextlib.suppress(Exception):
                                await exchange.close()
                        else:
                            with contextlib.suppress(Exception):
                                await asyncio.to_thread(exchange.close)
                    raise
                except Exception as exc:
                    WS_FAIL_COUNTS[symbol] = WS_FAIL_COUNTS.get(symbol, 0) + 1
                    if WS_FAIL_COUNTS[symbol] > 2:
                        logger.warning(
                            "WS OHLCV failed: %s - falling back to REST", exc
                        )
                        return await _fetch_ohlcv_async_inner(
                            exchange,
                            symbol,
                            timeframe=timeframe,
                            limit=limit,
                            since=since,
                            use_websocket=False,
                            force_websocket_history=force_websocket_history,
                        )
                    if attempt >= 2:
                        raise
                    logger.warning(
                        "watch_ohlcv failed on attempt %d: %s", attempt + 1, exc
                    )
                    await asyncio.sleep(5)
            if ws_limit and len(data) < ws_limit and force_websocket_history:
                logger.warning(
                    "WebSocket OHLCV for %s %s returned %d of %d candles; disable force_websocket_history to allow REST fallback",
                    symbol,
                    timeframe,
                    len(data),
                    ws_limit,
                )
            if (
                ws_limit
                and len(data) < ws_limit
                and not force_websocket_history
                and hasattr(exchange, "fetch_ohlcv")
            ):
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        data = await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                    expected = limit
                    if since is not None:
                        try:
                            tf_sec = timeframe_seconds(exchange, timeframe)
                            now_ms = int(time.time() * 1000)
                            expected = min(
                                limit, int((now_ms - since) // (tf_sec * 1000)) + 1
                            )
                        except Exception:
                            pass
                    if len(data) < expected:
                        logger.info(
                            "Incomplete OHLCV for %s: got %d of %d",
                            symbol,
                            len(data),
                            expected,
                        )
                    return data
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    data = await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
                expected = limit
                if since is not None:
                    try:
                        tf_sec = timeframe_seconds(exchange, timeframe)
                        now_ms = int(time.time() * 1000)
                        expected = min(
                            limit, int((now_ms - since) // (tf_sec * 1000)) + 1
                        )
                    except Exception:
                        pass
                if len(data) < expected:
                    logger.info(
                        "Incomplete OHLCV for %s: got %d of %d",
                        symbol,
                        len(data),
                        expected,
                    )
                return data
            expected = limit
            if since is not None:
                try:
                    tf_sec = timeframe_seconds(exchange, timeframe)
                    now_ms = int(time.time() * 1000)
                    expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
                except Exception:
                    pass
            if len(data) < expected:
                logger.info(
                    "Incomplete OHLCV for %s: got %d of %d",
                    symbol,
                    len(data),
                    expected,
                )
                if since is not None and hasattr(exchange, "fetch_ohlcv"):
                    try:
                        kwargs_r = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "limit": limit,
                        }
                        if asyncio.iscoroutinefunction(
                            getattr(exchange, "fetch_ohlcv", None)
                        ):
                            try:
                                data_r = await _call_with_retry(
                                    exchange.fetch_ohlcv,
                                    timeout=REST_OHLCV_TIMEOUT,
                                    **kwargs_r,
                                )
                            except asyncio.CancelledError:
                                raise
                        else:
                            try:
                                data_r = await _call_with_retry(
                                    asyncio.to_thread,
                                    exchange.fetch_ohlcv,
                                    **kwargs_r,
                                    timeout=REST_OHLCV_TIMEOUT,
                                )
                            except asyncio.CancelledError:
                                raise
                        if len(data_r) > len(data):
                            data = data_r
                    except Exception:
                        pass
                if (
                    len(data) < expected
                    and since is not None
                    and hasattr(exchange, "fetch_trades")
                ):
                    try:
                        trades_data = await fetch_ohlcv_from_trades(
                            exchange,
                            symbol,
                            timeframe,
                            since,
                            limit,
                        )
                        if len(trades_data) > len(data):
                            data = trades_data
                    except Exception:
                        pass
            return data
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
            params_f = inspect.signature(exchange.fetch_ohlcv).parameters
            kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
            if since is not None and "since" in params_f:
                kwargs_f["since"] = since
            try:
                data = await _call_with_retry(
                    exchange.fetch_ohlcv,
                    timeout=REST_OHLCV_TIMEOUT,
                    **kwargs_f,
                )
            except asyncio.CancelledError:
                raise
            expected = limit
            if since is not None:
                try:
                    tf_sec = timeframe_seconds(exchange, timeframe)
                    now_ms = int(time.time() * 1000)
                    expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
                except Exception:
                    pass
            if len(data) < expected:
                logger.info(
                    "Incomplete OHLCV for %s: got %d of %d",
                    symbol,
                    len(data),
                    expected,
                )
            if since is not None:
                try:
                    kwargs_r = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    try:
                        data_r = await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_r,
                        )
                    except asyncio.CancelledError:
                        raise
                    if len(data_r) > len(data):
                        data = data_r
                except Exception:
                    pass
            if (
                len(data) < expected
                and since is not None
                and hasattr(exchange, "fetch_trades")
            ):
                try:
                    trades_data = await fetch_ohlcv_from_trades(
                        exchange,
                        symbol,
                        timeframe,
                        since,
                        limit,
                    )
                    if len(trades_data) > len(data):
                        data = trades_data
                except Exception:
                    pass
            return data
        params_f = inspect.signature(exchange.fetch_ohlcv).parameters
        kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        if since is not None and "since" in params_f:
            kwargs_f["since"] = since
        try:
            data = await _call_with_retry(
                asyncio.to_thread,
                exchange.fetch_ohlcv,
                **kwargs_f,
                timeout=REST_OHLCV_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        expected = limit
        if since is not None:
            try:
                tf_sec = timeframe_seconds(exchange, timeframe)
                now_ms = int(time.time() * 1000)
                expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
            except Exception:
                pass
        if len(data) < expected:
            logger.info(
                "Incomplete OHLCV for %s: got %d of %d",
                symbol,
                len(data),
                expected,
            )
            if since is not None:
                try:
                    kwargs_r = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    try:
                        data_r = await _call_with_retry(
                            asyncio.to_thread,
                            exchange.fetch_ohlcv,
                            **kwargs_r,
                            timeout=REST_OHLCV_TIMEOUT,
                        )
                    except asyncio.CancelledError:
                        raise
                    if len(data_r) > len(data):
                        data = data_r
                except Exception:
                    pass
        if (
            len(data) < expected
            and since is not None
            and hasattr(exchange, "fetch_trades")
        ):
            try:
                trades_data = await fetch_ohlcv_from_trades(
                    exchange,
                    symbol,
                    timeframe,
                    since,
                    limit,
                )
                if len(trades_data) > len(data):
                    data = trades_data
            except Exception:
                pass
        return data
    except asyncio.TimeoutError as exc:
        ex_id = getattr(exchange, "id", "unknown")
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            logger.error(
                "WS OHLCV timeout for %s on %s (tf=%s limit=%s ws=%s): %s",
                symbol,
                ex_id,
                timeframe,
                limit,
                use_websocket,
                exc,
                exc_info=False,
            )
        else:
            logger.error(
                "REST OHLCV timeout for %s on %s (tf=%s limit=%s ws=%s): %s",
                symbol,
                ex_id,
                timeframe,
                limit,
                use_websocket,
                exc,
                exc_info=False,
            )
        if use_websocket and hasattr(exchange, "fetch_ohlcv"):
            logger.info(
                "Falling back to REST fetch_ohlcv for %s on %s limit %d",
                symbol,
                timeframe,
                limit,
            )
            try:
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        return await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    return await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
            except Exception as exc2:  # pragma: no cover - fallback
                ex_id = getattr(exchange, "id", "unknown")
                logger.error(
                    "REST fallback fetch_ohlcv failed for %s on %s (tf=%s limit=%s ws=%s): %s",
                    symbol,
                    ex_id,
                    timeframe,
                    limit,
                    use_websocket,
                    exc2,
                    exc_info=True,
                )
        return exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network
        if (
            use_websocket
            and hasattr(exchange, "fetch_ohlcv")
            and not force_websocket_history
        ):
            try:
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        return await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    return await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
            except Exception:
                pass
        return exc


async def fetch_ohlcv_async(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> list | Exception:
    """Return OHLCV data for ``symbol`` with simple retries."""

    for attempt in range(3):
        try:
            return await _fetch_ohlcv_async_inner(
                exchange,
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since,
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
            )
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "OHLCV fetch retry %d for %s: %s",
                attempt + 1,
                symbol,
                exc,
            )
            await asyncio.sleep(2 ** attempt)
    return []


async def fetch_geckoterminal_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    *,
    min_24h_volume: float = 0.0,
    return_price: bool = False,
) -> tuple[list, float] | tuple[list, float, float] | None:
    """Return OHLCV data and 24h volume for ``symbol`` from GeckoTerminal.

    When ``return_price`` is ``True`` the pool price is returned instead of the
    reserve liquidity value.
    """

    from urllib.parse import quote_plus

    async with GECKO_SEMAPHORE:
        # Validate symbol before making any requests
        try:
            token_mint, quote = symbol.split("/", 1)
        except ValueError:
            token_mint, quote = symbol, ""
        if quote != "USDC" or not _is_valid_base_token(token_mint):
            return None

        if symbol in GECKO_UNAVAILABLE:
            return None

        cached = GECKO_POOL_CACHE.get(symbol)
        is_cached = cached is not None and cached[4] == limit
        if not _is_valid_base_token(token_mint):
            return None

        volume = 0.0
        reserve = 0.0
        price = 0.0
        data: dict | None = None

        pool_addr = ""
        attrs: dict = {}
        if is_cached:
            pool_addr, volume, reserve, price, _ = cached

        backoff = 1
        for attempt in range(3):
            try:
                if not is_cached:
                    query = quote_plus(symbol)
                    search_url = "https://api.geckoterminal.com/api/v2/search/pools"
                    params = {"query": query, "network": "solana"}
                    search_data = await gecko_request(search_url, params=params)
                    if not search_data:
                        logger.info("token not available on GeckoTerminal: %s", symbol)
                        logger.info("pair not available on GeckoTerminal: %s", symbol)
                        GECKO_UNAVAILABLE.add(symbol)
                        return None

                    items = search_data.get("data") or []
                    if not items:
                        mint = await get_mint_from_gecko(token_mint)
                        if mint and mint != token_mint:
                            params["query"] = quote_plus(f"{mint}/USDC")
                            search_data = await gecko_request(search_url, params=params)
                            items = search_data.get("data") or [] if search_data else []
                        if mint:
                            params = {"query": mint, "network": "solana"}
                            search_data = await gecko_request(search_url, params=params)
                            items = search_data.get("data") or [] if search_data else []
                            token_mint = mint
                        if not items:
                            logger.info("pair not available on GeckoTerminal: %s", symbol)
                            GECKO_UNAVAILABLE.add(symbol)
                            return None

                    first = items[0]
                    attrs = first.get("attributes", {}) if isinstance(first, dict) else {}
                    if not attrs:
                        helius_map = await fetch_from_helius([token_mint])
                        helius_mint = helius_map.get(token_mint.upper()) if isinstance(helius_map, dict) else None
                        if helius_mint:
                            logger.info("Helius mint resolved for %s: %s", symbol, helius_mint)
                            params = {"query": helius_mint, "network": "solana"}
                            search_data = await gecko_request(search_url, params=params)
                            items = search_data.get("data") or [] if search_data else []
                            if items:
                                first = items[0]
                                attrs = first.get("attributes", {}) if isinstance(first, dict) else {}
                                token_mint = helius_mint
                        if not attrs:
                            return None
                    
                    pool_id = str(first.get("id", ""))
                    pool_addr = pool_id.split("_", 1)[-1]
                    try:
                        volume = float(attrs.get("volume_usd", {}).get("h24", 0.0))
                    except Exception:
                        volume = 0.0
                    if volume < float(min_24h_volume):
                        return None
                    try:
                        price = float(attrs.get("base_token_price_quote_token", 0.0))
                    except Exception:
                        price = 0.0
                    try:
                        reserve = float(attrs.get("reserve_in_usd", 0.0))
                    except Exception:
                        reserve = 0.0

                ohlcv_url = (
                    f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_addr}/ohlcv/{timeframe}"
                )
                params = {"aggregate": 1, "limit": limit}
                data = await gecko_request(ohlcv_url, params=params)
                if data is None:
                    raise RuntimeError("request failed")
                break
            except Exception as exc:  # pragma: no cover - network
                if attempt == 2:
                    logger.error("GeckoTerminal OHLCV error for %s: %s", symbol, exc)
                    return None
                await asyncio.sleep(backoff)
                backoff = min(backoff + 1, 3)

        candles = (data.get("data") or {}).get("attributes", {}).get("ohlcv_list") or []

        result: list = []
        multiplier = 1000 if is_cached else 1
        for c in candles[-limit:]:
            try:
                result.append(
                    [
                        int(c[0]) * multiplier,
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]),
                    ]
                )
            except Exception:
                reserve = 0.0
        GECKO_POOL_CACHE[symbol] = (pool_addr, volume, reserve, price, limit)

        if return_price:
            return result, volume, price
        return result, volume, reserve


async def fetch_coingecko_ohlc(
    coin_id: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> list | None:
    """Return OHLC data from CoinGecko as [timestamp, open, high, low, close, 0]."""

    days = 1
    if timeframe.endswith("d"):
        days = 90
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception:  # pragma: no cover - network
        return None

    result: list = []
    for c in data[-limit:]:
        if not isinstance(c, list) or len(c) < 5:
            continue
        try:
            ts, o, h, l, cl = c[:5]
            result.append([int(ts), float(o), float(h), float(l), float(cl), 0.0])
        except Exception:
            continue
    return result


async def get_kraken_listing_date(symbol: str) -> int | None:
    """Return approximate Kraken listing date in milliseconds for ``symbol``."""
    if symbol in KRAKEN_LISTING_CACHE:
        return KRAKEN_LISTING_CACHE[symbol]

    url = "https://api.kraken.com/0/public/Trades"
    ex = ccxt.kraken()
    try:
        pair = ex.market_id(symbol)
    except Exception:
        pair = symbol.replace("/", "")
    params = {"pair": pair, "since": 0}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception:
        KRAKEN_LISTING_CACHE[symbol] = None
        return None
    trades = None
    if isinstance(data, dict):
        for val in data.get("result", {}).values():
            if isinstance(val, list) and val:
                trades = val
                break
    if not trades:
        KRAKEN_LISTING_CACHE[symbol] = None
        return None
    try:
        ts = float(trades[0][2])
        dt = datetime.datetime.utcfromtimestamp(ts)
        ts_ms = int(dt.timestamp() * 1000)
        KRAKEN_LISTING_CACHE[symbol] = ts_ms
        return ts_ms
    except Exception:
        KRAKEN_LISTING_CACHE[symbol] = None
        return None


async def fetch_dex_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    *,
    min_volume_usd: float = 0.0,
    gecko_res: list | tuple | None = None,
    use_gecko: bool = True,
) -> list | None:
    """Fetch DEX OHLCV with fallback to CoinGecko, Coinbase then Kraken."""

    res = gecko_res
    if res is None and use_gecko:
        try:
            res = await fetch_geckoterminal_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # pragma: no cover - network
            logger.error("GeckoTerminal OHLCV error for %s: %s", symbol, exc)
            res = None

    data = None
    if res:
        if isinstance(res, tuple):
            data, vol = res
        else:
            data = res
            vol = min_volume_usd
        if data and vol >= min_volume_usd:
            return data

    base, _, quote = symbol.partition("/")
    coin_id = COINGECKO_IDS.get(base)
    if coin_id:
        data = await fetch_coingecko_ohlc(coin_id, timeframe=timeframe, limit=limit)
        if data:
            return data

    if quote.upper() in SUPPORTED_USD_QUOTES:
        try:
            cb = ccxt.coinbase({"enableRateLimit": True})
            data = await fetch_ohlcv_async(cb, symbol, timeframe=timeframe, limit=limit)
        finally:
            close = getattr(cb, "close", None)
            if close:
                try:
                    if asyncio.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
                except Exception:
                    pass
        if data and not isinstance(data, Exception):
            return data

    data = await fetch_ohlcv_async(exchange, symbol, timeframe=timeframe, limit=limit)
    if isinstance(data, Exception):
        return None
    return data


async def fetch_ohlcv_from_trades(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    since: int | None = None,
    limit: int = 100,
) -> list:
    """Aggregate trades into OHLCV format."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchTrades"):
        return []

    fetch_fn = getattr(exchange, "fetch_trades", None)
    if fetch_fn is None:
        return []

    params = inspect.signature(fetch_fn).parameters
    kwargs = {"symbol": symbol, "limit": limit * 100}
    if since is not None and "since" in params:
        kwargs["since"] = since

    try:
        if asyncio.iscoroutinefunction(fetch_fn):
            trades = await _call_with_retry(
                fetch_fn, timeout=REST_OHLCV_TIMEOUT, **kwargs
            )
        else:
            trades = await _call_with_retry(
                asyncio.to_thread,
                fetch_fn,
                **kwargs,
                timeout=REST_OHLCV_TIMEOUT,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        return []

    if not trades:
        return []

    tf_ms = timeframe_seconds(exchange, timeframe) * 1000
    trades.sort(key=lambda t: t[0])

    ohlcv: list[list] = []
    bucket = trades[0][0] - trades[0][0] % tf_ms
    o = h = l = c = float(trades[0][1])
    vol = float(trades[0][2]) if len(trades[0]) > 2 else 0.0

    for t in trades[1:]:
        ts = int(t[0])
        price = float(t[1])
        amount = float(t[2]) if len(t) > 2 else 0.0
        b = ts - ts % tf_ms
        if b != bucket:
            ohlcv.append([bucket, o, h, l, c, vol])
            if len(ohlcv) >= limit:
                return ohlcv[:limit]
            bucket = b
            o = h = l = c = price
            vol = amount
        else:
            h = max(h, price)
            l = min(l, price)
            c = price
            vol += amount

    ohlcv.append([bucket, o, h, l, c, vol])
    return ohlcv[:limit]


async def fetch_order_book_async(
    exchange,
    symbol: str,
    depth: int = 2,
) -> dict | Exception:
    """Return order book snapshot for ``symbol`` with top ``depth`` levels."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchOrderBook"):
        return {}

    try:
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_order_book", None)):
            return await asyncio.wait_for(
                exchange.fetch_order_book(symbol, limit=depth), OHLCV_TIMEOUT
            )
        return await asyncio.wait_for(
            asyncio.to_thread(exchange.fetch_order_book, symbol, depth),
            OHLCV_TIMEOUT,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network
        return exc


async def load_ohlcv_parallel(
    exchange,
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    since_map: Dict[str, int] | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
) -> Dict[str, list]:
    """Fetch OHLCV data for multiple symbols concurrently.

    Parameters
    ----------
    notifier : TelegramNotifier | None, optional
        If provided, failures will be sent using this notifier.
    """

    since_map = since_map or {}

    now = time.time()
    filtered_symbols: List[str] = []
    for s in symbols:
        info = failed_symbols.get(s)
        if not info:
            filtered_symbols.append(s)
            continue
        if info.get("disabled"):
            continue
        if now - info["time"] >= info["delay"]:
            filtered_symbols.append(s)
    symbols = filtered_symbols

    if not symbols:
        return {}

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")
        sem = asyncio.Semaphore(max_concurrent)
    elif SEMA is not None:
        sem = SEMA
    else:
        sem = None

    async def sem_fetch(sym: str):
        async def _fetch_and_sleep():
            data = await fetch_ohlcv_async(
                exchange,
                sym,
                timeframe=timeframe,
                limit=limit,
                since=since_map.get(sym),
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
            )
            rl = getattr(exchange, "rateLimit", None)
            if rl:
                await asyncio.sleep(rl / 1000)
            await asyncio.sleep(1)
            return data

        if sem:
            async with sem:
                return await _fetch_and_sleep()

        return await _fetch_and_sleep()

    tasks = [asyncio.create_task(sem_fetch(s)) for s in symbols]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    if any(isinstance(r, asyncio.CancelledError) for r in results):
        for t in tasks:
            if not t.done():
                t.cancel()
        raise asyncio.CancelledError()

    data: Dict[str, list] = {}
    ex_id = getattr(exchange, "id", "unknown")
    mode = "websocket" if use_websocket else "REST"
    for sym, res in zip(symbols, results):
        if isinstance(res, asyncio.CancelledError):
            raise res
        if isinstance(res, asyncio.TimeoutError):
            logger.error(
                "Timeout loading OHLCV for %s on %s limit %d: %s",
                sym,
                timeframe,
                limit,
                res,
                exc_info=True,
            )
            msg = (
                f"Timeout loading OHLCV for {sym} on {ex_id} "
                f"(tf={timeframe} limit={limit} mode={mode})"
            )
            logger.error(msg)
            if notifier and STATUS_UPDATES:
                notifier.notify(
                    f"Timeout loading OHLCV for {sym} on {timeframe} limit {limit}"
                )
            info = failed_symbols.get(sym)
            delay = RETRY_DELAY
            count = 1
            disabled = False
            if info is not None:
                delay = min(info["delay"] * 2, MAX_RETRY_DELAY)
                count = info.get("count", 0) + 1
                disabled = info.get("disabled", False)
            if count >= MAX_OHLCV_FAILURES:
                disabled = True
                if not info or not info.get("disabled"):
                    logger.info("Disabling %s after %d OHLCV failures", sym, count)
            failed_symbols[sym] = {
                "time": time.time(),
                "delay": delay,
                "count": count,
                "disabled": disabled,
            }
            continue
        if (
            isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError)
        ) or not res:
            logger.error(
                "Failed to load OHLCV for %s on %s limit %d: %s",
                sym,
                timeframe,
                limit,
                res,
                exc_info=isinstance(res, Exception),
            )
            msg = (
                f"Failed to load OHLCV for {sym} on {ex_id} "
                f"(tf={timeframe} limit={limit} mode={mode}): {res}"
            )
            logger.error(msg)
            if notifier and STATUS_UPDATES:
                notifier.notify(
                    f"Failed to load OHLCV for {sym} on {timeframe} limit {limit}: {res}"
                )
            info = failed_symbols.get(sym)
            delay = RETRY_DELAY
            count = 1
            disabled = False
            if info is not None:
                delay = min(info["delay"] * 2, MAX_RETRY_DELAY)
                count = info.get("count", 0) + 1
                disabled = info.get("disabled", False)
            if count >= MAX_OHLCV_FAILURES:
                disabled = True
                if not info or not info.get("disabled"):
                    logger.info("Disabling %s after %d OHLCV failures", sym, count)
            failed_symbols[sym] = {
                "time": time.time(),
                "delay": delay,
                "count": count,
                "disabled": disabled,
            }
            continue
        if res and len(res[0]) > 6:
            res = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in res]
        data[sym] = res
        failed_symbols.pop(sym, None)
    return data


async def _update_ohlcv_cache_inner(
    exchange,
    cache: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    start_since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    config: Dict | None = None,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
) -> Dict[str, pd.DataFrame]:
    """Update cached OHLCV DataFrames with new candles.

    Parameters
    ----------
    max_concurrent : int | None, optional
        Maximum number of concurrent OHLCV requests. ``None`` means no limit.
    start_since : int | None, optional
        When provided, fetch data starting from this timestamp in milliseconds.
    """

    from crypto_bot.regime.regime_classifier import clear_regime_cache

    # Use the provided limit without enforcing a fixed minimum
    limit = int(limit)
    # Request the number of candles specified by the caller

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")

    global _last_snapshot_time
    config = config or {}
    snapshot_interval = config.get("ohlcv_snapshot_frequency_minutes", 1440) * 60
    now = time.time()
    snapshot_due = now - _last_snapshot_time >= snapshot_interval

    logger.info("Starting OHLCV update for timeframe %s", timeframe)

    since_map: Dict[str, int | None] = {}
    if start_since is not None:
        tf_sec = timeframe_seconds(exchange, timeframe)
        needed = int((time.time() * 1000 - start_since) // (tf_sec * 1000)) + 1
        limit = max(limit, needed)
        since_map = {sym: start_since for sym in symbols}
        snapshot_due = False
    elif snapshot_due:
        _last_snapshot_time = now
        limit = max(config.get("ohlcv_snapshot_limit", limit), limit)
        since_map = {sym: None for sym in symbols}
    else:
        for sym in symbols:
            df = cache.get(sym)
            if df is not None and not df.empty:
                # convert cached second timestamps to milliseconds for ccxt
                since_map[sym] = int(df["timestamp"].iloc[-1]) * 1000 + 1
            elif start_since is not None:
                since_map[sym] = start_since
    now = time.time()
    filtered_symbols: List[str] = []
    for s in symbols:
        info = failed_symbols.get(s)
        if not info:
            filtered_symbols.append(s)
            continue
        if info.get("disabled"):
            continue
        if now - info["time"] >= info["delay"]:
            filtered_symbols.append(s)
    symbols = filtered_symbols
    if not symbols:
        return cache

    logger.info(
        "Fetching %d candles for %d symbols on %s",
        limit,
        len(symbols),
        timeframe,
    )

    data_map: Dict[str, list] = {s: [] for s in symbols}
    remaining = limit
    curr_since = since_map.copy()
    while remaining > 0:
        req_limit = min(remaining, 720)
        batch = await load_ohlcv_parallel(
            exchange,
            symbols,
            timeframe,
            req_limit,
            curr_since,
            use_websocket,
            force_websocket_history,
            max_concurrent,
            notifier,
        )
        for sym, rows in batch.items():
            if rows:
                data_map[sym].extend(rows)
                last_ts = rows[-1][0]
                curr_since[sym] = last_ts + timeframe_seconds(exchange, timeframe) * 1000
        if all(len(batch.get(sym, [])) < req_limit for sym in symbols):
            break
        remaining -= req_limit

    logger.info(
        "Fetched OHLCV for %d/%d symbols on %s",
        len([s for s in symbols if s in data_map]),
        len(symbols),
        timeframe,
    )

    for sym in symbols:
        data = data_map.get(sym)
        if not data:
            info = failed_symbols.get(sym)
            skip_retry = (
                info is not None
                and time.time() - info["time"] < info["delay"]
                and since_map.get(sym) is None
            )
            if skip_retry:
                continue
            failed_symbols.pop(sym, None)
            full = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit,
                None,
                use_websocket,
                force_websocket_history,
                max_concurrent,
                notifier,
            )
            data = full.get(sym)
            if data:
                failed_symbols.pop(sym, None)
        if data is None:
            continue
        df_new = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        tf_sec = timeframe_seconds(None, timeframe)
        unit = "ms" if df_new["timestamp"].iloc[0] > 1e10 else "s"
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit=unit)
        df_new = (
            df_new.set_index("timestamp")
            .resample(f"{tf_sec}s")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .ffill()
            .reset_index()
        )
        df_new["timestamp"] = df_new["timestamp"].astype(int) // 10 ** 9
        frac = config.get("min_history_fraction", 0.5)
        try:
            frac_val = float(frac)
        except (TypeError, ValueError):
            frac_val = 0.5
        min_candles_required = int(limit * frac_val)
        if len(df_new) < min_candles_required:
            since_val = since_map.get(sym)
            retry = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit * 2,
                {sym: since_val},
                False,
                force_websocket_history,
                max_concurrent,
                notifier,
            )
            retry_data = retry.get(sym)
            if retry_data and len(retry_data) > len(data):
                data = retry_data
                df_new = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
            if len(df_new) < min_candles_required:
                logger.warning(
                    "Skipping %s: only %d/%d candles",
                    sym,
                    len(df_new),
                    limit,
                )
                continue
        changed = False
        if sym in cache and not cache[sym].empty:
            last_ts = cache[sym]["timestamp"].iloc[-1]
            df_new = df_new[df_new["timestamp"] > last_ts]
            if df_new.empty:
                continue
            cache[sym] = pd.concat([cache[sym], df_new], ignore_index=True)
            changed = True
        else:
            cache[sym] = df_new
            changed = True
        if changed:
            cache[sym] = cache[sym].tail(limit).reset_index(drop=True)
            cache[sym]["return"] = cache[sym]["close"].pct_change()
            clear_regime_cache(sym, timeframe)
    logger.info("Completed OHLCV update for timeframe %s", timeframe)
    return cache


async def update_ohlcv_cache(
    exchange,
    cache: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    start_since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    config: Dict | None = None,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
    batch_size: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Batch OHLCV updates for multiple calls."""

    config = config or {}
    delay = 0.5
    size = (
        batch_size
        if batch_size is not None
        else config.get("ohlcv_batch_size", 3)
    )
    if size is None:
        cfg_size = config.get("ohlcv_batch_size")
        if isinstance(batch_size, int) and batch_size >= 1:
            size = batch_size
        elif isinstance(cfg_size, int) and cfg_size >= 1:
            size = cfg_size
        else:
            if batch_size is not None or cfg_size is not None:
                logger.warning(
                    "Invalid ohlcv_batch_size %r; defaulting to 3",
                    batch_size if batch_size is not None else cfg_size,
                )
            else:
                logger.warning("ohlcv_batch_size not set; defaulting to 3")
            size = 3
    key = (
        timeframe,
        limit,
        start_since,
        use_websocket,
        force_websocket_history,
        max_concurrent,
    )

    req = _OhlcvBatchRequest(
        exchange,
        cache,
        list(symbols),
        timeframe,
        limit,
        start_since,
        use_websocket,
        force_websocket_history,
        config,
        max_concurrent,
        notifier,
        asyncio.get_running_loop().create_future(),
    )

    queue = _OHLCV_BATCH_QUEUES.setdefault(key, asyncio.Queue())
    await queue.put(req)

    if key not in _OHLCV_BATCH_TASKS or _OHLCV_BATCH_TASKS[key].done():
        _OHLCV_BATCH_TASKS[key] = asyncio.create_task(
            _ohlcv_batch_worker(key, queue, size, delay)
        )

    return await req.future


async def update_multi_tf_ohlcv_cache(
    exchange,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    symbols: Iterable[str],
    config: Dict,
    limit: int = 100,
    start_since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
    priority_queue: Deque[str] | None = None,
    batch_size: int | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for multiple timeframes.

    Parameters
    ----------
    config : Dict
        Configuration containing a ``timeframes`` list.
    start_since : int | None, optional
        When provided, fetch historical data starting from this timestamp
        in milliseconds when no cached data is available.
    """
    from crypto_bot.regime.regime_classifier import clear_regime_cache

    limit = int(limit)
    # Use the limit provided by the caller

    def add_priority(data: list, symbol: str) -> None:
        """Push ``symbol`` to ``priority_queue`` if volume spike detected."""
        if priority_queue is None or vol_thresh is None or not data:
            return
        try:
            vols = np.array([row[5] for row in data], dtype=float)
            mean = float(np.mean(vols)) if len(vols) else 0.0
            std = float(np.std(vols))
            if std <= 0:
                return
            z_max = float(np.max((vols - mean) / std))
            if z_max > vol_thresh:
                priority_queue.appendleft(symbol)
        except Exception:
            return

    tfs = config.get("timeframes", ["1h"])
    supported = getattr(exchange, "timeframes", None)
    if supported:
        unsupported = [tf for tf in tfs if tf not in supported]
        if unsupported:
            logger.info(
                "Skipping unsupported timeframes on %s: %s",
                getattr(exchange, "id", "unknown"),
                unsupported,
            )
        tfs = [tf for tf in tfs if tf in supported]
    logger.info("Updating OHLCV cache for timeframes: %s", tfs)
    if not tfs:
        return cache

    min_volume_usd = float(config.get("min_volume_usd", 0) or 0)
    vol_thresh = config.get("bounce_scalper", {}).get("vol_zscore_threshold")

    for tf in tfs:
        logger.info("Starting update for timeframe %s", tf)
        tf_cache = cache.get(tf, {})

        now_ms = int(time.time() * 1000)
        dynamic_limits: dict[str, int] = {}
        snapshot_cap = int(config.get("ohlcv_snapshot_limit", limit))
        max_cap = min(snapshot_cap, 720)

        concurrency = int(config.get("listing_date_concurrency", 5) or 0)
        semaphore = asyncio.Semaphore(concurrency) if concurrency > 0 else None

        async def _fetch_listing(sym: str) -> tuple[str, int | None]:
            if semaphore is not None:
                async with semaphore:
                    ts = await get_kraken_listing_date(sym)
            else:
                ts = await get_kraken_listing_date(sym)
            return sym, ts

        tasks = [asyncio.create_task(_fetch_listing(sym)) for sym in symbols]
        for sym, listing_ts in await asyncio.gather(*tasks):
            if listing_ts and 0 < listing_ts <= now_ms:
                age_ms = now_ms - listing_ts
                tf_sec = timeframe_seconds(exchange, tf)
                hist_candles = age_ms // (tf_sec * 1000)
                if hist_candles > 0:
                    dynamic_limits[sym] = int(min(hist_candles, max_cap))

        cex_symbols: list[str] = []
        dex_symbols: list[str] = []
        for s in symbols:
            sym = s
            base, _, quote = s.partition("/")
            is_solana = quote.upper() == "USDC" and base.upper() not in NON_SOLANA_BASES
            if is_solana:
                dex_symbols.append(sym)
            else:
                if "coinbase" in getattr(exchange, "id", "") and "/USDC" in sym:
                    mapped = sym.replace("/USDC", "/USD")
                    if mapped not in getattr(exchange, "symbols", []):
                        continue  # skip unsupported pair
                    sym = mapped
                cex_symbols.append(sym)

        tf_sec = timeframe_seconds(exchange, tf)
        tf_limit = limit
        if start_since is not None:
            needed = int((time.time() * 1000 - start_since) // (tf_sec * 1000)) + 1
            tf_limit = max(limit, needed)

        if cex_symbols and start_since is None:
            groups: Dict[int, list[str]] = {}
            for sym in cex_symbols:
                sym_limit = dynamic_limits.get(sym, tf_limit)
                groups.setdefault(int(sym_limit), []).append(sym)
            for lim, syms in groups.items():
                if lim < tf_limit:
                    for s in syms:
                        logger.info(
                            "Adjusting limit for %s on %s to %d", s, tf, lim
                        )
                    tf_cache = await update_ohlcv_cache(
                        exchange,
                        tf_cache,
                        syms,
                        timeframe=tf,
                        limit=lim,
                        config={
                            "min_history_fraction": 0,
                            "ohlcv_batch_size": config.get("ohlcv_batch_size"),
                        },
                        batch_size=batch_size,
                        start_since=start_since,
                        use_websocket=use_websocket,
                        force_websocket_history=force_websocket_history,
                        max_concurrent=max_concurrent,
                        notifier=notifier,
                )
        elif cex_symbols:
            from crypto_bot.main import update_df_cache

            for sym in cex_symbols:
                batches: list = []
                current_since = start_since
                sym_total = min(tf_limit, dynamic_limits.get(sym, tf_limit))
                if sym_total < tf_limit:
                    logger.info(
                        "Adjusting limit for %s on %s to %d", sym, tf, sym_total
                    )
                remaining = sym_total
                while remaining > 0:
                    req = min(remaining, 1000)
                    data = await fetch_ohlcv_async(
                        exchange,
                        sym,
                        timeframe=tf,
                        limit=req,
                        since=current_since,
                        use_websocket=use_websocket,
                        force_websocket_history=force_websocket_history,
                    )
                    if not data or isinstance(data, Exception):
                        break
                    batches.extend(data)
                    remaining -= len(data)
                    if len(data) < req:
                        break
                    current_since = data[-1][0] + 1

                if not batches:
                    continue

                df_new = pd.DataFrame(
                    batches,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                tf_sec = timeframe_seconds(None, tf)
                unit = "ms" if df_new["timestamp"].iloc[0] > 1e10 else "s"
                df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit=unit)
                df_new = (
                    df_new.set_index("timestamp")
                    .resample(f"{tf_sec}s")
                    .agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    })
                    .ffill()
                    .reset_index()
                )
                df_new["timestamp"] = df_new["timestamp"].astype(int) // 10 ** 9

                if sym in tf_cache and not tf_cache[sym].empty:
                    last_ts = tf_cache[sym]["timestamp"].iloc[-1]
                    df_new = df_new[df_new["timestamp"] > last_ts]
                    if df_new.empty:
                        continue
                    df_new = pd.concat([tf_cache[sym], df_new], ignore_index=True)

                update_df_cache(cache, tf, sym, df_new)
                tf_cache = cache.get(tf, {})
                tf_cache[sym]["return"] = tf_cache[sym]["close"].pct_change()
                clear_regime_cache(sym, tf)

        for sym in dex_symbols:
            data = None
            vol = 0.0
            res = None
            gecko_failed = False
            base, _, quote = sym.partition("/")
            is_solana = quote.upper() == "USDC" and base.upper() not in NON_SOLANA_BASES
            sym_l = min(dynamic_limits.get(sym, tf_limit), tf_limit)
            if sym_l < tf_limit:
                logger.info("Adjusting limit for %s on %s to %d", sym, tf, sym_l)
            if is_solana:
                try:
                    res = await fetch_geckoterminal_ohlcv(
                        sym,
                        timeframe=tf,
                        limit=sym_l,
                        min_24h_volume=min_volume_usd,
                    )
                except Exception as e:  # pragma: no cover - network
                    logger.warning(
                        f"Gecko failed for {sym}: {e} - using exchange data"
                    )
                    gecko_failed = True
            else:
                gecko_failed = True

            if res and not gecko_failed:
                if isinstance(res, tuple):
                    data, vol, *_ = res
                else:
                    data = res
                    vol = min_volume_usd
                add_priority(data, sym)

            if gecko_failed or not data or vol < min_volume_usd:
                data = await fetch_dex_ohlcv(
                    exchange,
                    sym,
                    timeframe=tf,
                    limit=sym_l,
                    min_volume_usd=min_volume_usd,
                    gecko_res=res,
                    use_gecko=is_solana,
                )
                if isinstance(data, Exception) or not data:
                    continue

            if not data:
                continue

            if not isinstance(data, list):
                logger.error(
                    "Invalid OHLCV data type for %s on %s (type: %s), skipping",
                    sym,
                    tf,
                    type(data),
                )
                continue

            df_new = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            changed = False
            if sym in tf_cache and not tf_cache[sym].empty:
                last_ts = tf_cache[sym]["timestamp"].iloc[-1]
                df_new = df_new[df_new["timestamp"] > last_ts]
                if df_new.empty:
                    continue
                tf_cache[sym] = pd.concat([tf_cache[sym], df_new], ignore_index=True)
                changed = True
            else:
                tf_cache[sym] = df_new
                changed = True
            if changed:
                tf_cache[sym]["return"] = tf_cache[sym]["close"].pct_change()
                clear_regime_cache(sym, tf)

        cache[tf] = tf_cache
        logger.info("Finished update for timeframe %s", tf)

    return cache


async def update_regime_tf_cache(
    exchange,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    symbols: Iterable[str],
    config: Dict,
    limit: int = 100,
    start_since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
    df_map: Dict[str, Dict[str, pd.DataFrame]] | None = None,
    batch_size: int | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for regime detection timeframes."""
    limit = int(limit)
    # Respect the caller-specified limit
    regime_cfg = {**config, "timeframes": config.get("regime_timeframes", [])}
    tfs = regime_cfg["timeframes"]
    logger.info("Updating regime cache for timeframes: %s", tfs)

    missing_tfs: List[str] = []
    if df_map is not None:
        for tf in tfs:
            tf_data = df_map.get(tf)
            if tf_data is None:
                missing_tfs.append(tf)
                continue
            tf_cache = cache.setdefault(tf, {})
            for sym in symbols:
                df = tf_data.get(sym)
                if df is not None:
                    tf_cache[sym] = df
            cache[tf] = tf_cache
    else:
        missing_tfs = tfs

    if missing_tfs:
        fetch_cfg = {**regime_cfg, "timeframes": missing_tfs}
        cache = await update_multi_tf_ohlcv_cache(
            exchange,
            cache,
            symbols,
            fetch_cfg,
            limit=limit,
            start_since=start_since,
            use_websocket=use_websocket,
            force_websocket_history=force_websocket_history,
            max_concurrent=max_concurrent,
            notifier=notifier,
            priority_queue=None,
            batch_size=batch_size,
        )

    return cache

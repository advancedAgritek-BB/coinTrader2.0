"""Utilities for loading trading symbols and fetching OHLCV data."""

from typing import Iterable, List, Dict, Any, Deque
import asyncio
import inspect
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import ccxt
import aiohttp
import base58
import warnings
import contextlib

from .telegram import TelegramNotifier
from .logger import LOG_DIR, setup_logger


_last_snapshot_time = 0

logger = setup_logger(__name__, LOG_DIR / "bot.log")

failed_symbols: Dict[str, Dict[str, Any]] = {}
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
UNSUPPORTED_SYMBOL = object()
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
GECKO_SEMAPHORE = asyncio.Semaphore(25)

# Valid characters for Solana addresses
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Quote currencies eligible for Coinbase fallback
SUPPORTED_USD_QUOTES = {"USD", "USDC", "USDT"}


def _is_valid_base_token(token: str) -> bool:
    """Return True if ``token`` looks like a Solana mint address."""
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
    """Call ``func`` with exponential back-off on 520/522 errors."""

    attempts = 3
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
                await asyncio.sleep(2**attempt)
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


async def fetch_ohlcv_async(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> list | Exception:
    """Return OHLCV data for ``symbol`` using async I/O."""

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
                failed_symbols[symbol] = {
                    "time": time.time(),
                    "delay": MAX_RETRY_DELAY,
                    "count": MAX_OHLCV_FAILURES,
                    "disabled": True,
                }
                return UNSUPPORTED_SYMBOL
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
            try:
                data = await _call_with_retry(
                    exchange.watch_ohlcv, timeout=WS_OHLCV_TIMEOUT, **kwargs
                )
            except asyncio.CancelledError:
                if hasattr(exchange, "close"):
                    if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                        with contextlib.suppress(Exception):
                            await exchange.close()
                    else:
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(exchange.close)
                raise
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
                        logger.warning(
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
                    logger.warning(
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
                logger.warning(
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
                logger.warning(
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
            logger.warning(
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


async def fetch_dexscreener_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> list | None:
    """Deprecated: use :func:`fetch_geckoterminal_ohlcv` instead."""

    warnings.warn(
        "fetch_dexscreener_ohlcv is deprecated; use fetch_geckoterminal_ohlcv",
        DeprecationWarning,
        stacklevel=2,
    )
    return await fetch_geckoterminal_ohlcv(symbol, timeframe=timeframe, limit=limit)


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
        if quote != "USDC":
            return None
        try:
            base58.b58decode(token_mint)
        except Exception:
            return None
    # Validate symbol before making any requests
    try:
        token_mint, quote = symbol.split("/", 1)
    except ValueError:
        token_mint, quote = symbol, ""
    if quote != "USDC":
        return None
    if not _is_valid_base_token(token_mint):
        return None

    volume = 0.0
    reserve = 0.0
    price = 0.0
    data = {}
    is_cached = False

    backoff = 1
    for attempt in range(3):
        cached = GECKO_POOL_CACHE.get(symbol)
        is_cached = cached is not None and cached[4] == limit
        try:
            async with aiohttp.ClientSession() as session:
                if cached is None:
                    query = quote_plus(symbol)
                    search_url = (
                        "https://api.geckoterminal.com/api/v2/search/pools"
                        f"?query={query}&network=solana"
                    )

                    async with session.get(search_url, timeout=10) as resp:
                        if resp.status == 404:
                            logger.info(
                                "pair not available on GeckoTerminal: %s", symbol
                            )
                            return None
                        resp.raise_for_status()
                        search_data = await resp.json()

                    items = search_data.get("data") or []
                    if not items:
                        logger.info("pair not available on GeckoTerminal: %s", symbol)
                        return None

                    first = items[0]
                    attrs = (
                        first.get("attributes", {}) if isinstance(first, dict) else {}
                    )

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

                    GECKO_POOL_CACHE[symbol] = (
                        pool_addr,
                        volume,
                        reserve,
                        price,
                        limit,
                    )
                else:
                    pool_addr, volume, reserve, price, _ = cached

                ohlcv_url = (
                    "https://api.geckoterminal.com/api/v2/networks/solana/pools/"
                    f"{pool_addr}/ohlcv/{timeframe}?aggregate=1&limit={limit}"
                )

                async with session.get(ohlcv_url, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            break
        except Exception as exc:  # pragma: no cover - network
            if attempt == 2:
                logger.error("GeckoTerminal OHLCV error for %s: %s", symbol, exc)
                return None
            await asyncio.sleep(backoff)
            backoff *= 2

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
            continue

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
        if res is UNSUPPORTED_SYMBOL:
            continue
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


async def update_ohlcv_cache(
    exchange,
    cache: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
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
    """

    from crypto_bot.regime.regime_classifier import clear_regime_cache

    # Ensure we always request a reasonable number of candles
    limit = max(limit, 200)

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
    if snapshot_due:
        _last_snapshot_time = now
        limit = max(config.get("ohlcv_snapshot_limit", limit), 200)
        since_map = {sym: None for sym in symbols}
    else:
        for sym in symbols:
            df = cache.get(sym)
            if df is not None and not df.empty:
                since_map[sym] = int(df["timestamp"].iloc[-1]) + 1
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

    data_map = await load_ohlcv_parallel(
        exchange,
        symbols,
        timeframe,
        limit,
        since_map,
        use_websocket,
        force_websocket_history,
        max_concurrent,
        notifier,
    )

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
        min_candles_required = int(limit * 0.5)
        if len(df_new) < min_candles_required:
            since_val = since_map.get(sym)
            retry = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit,
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
            cache[sym]["return"] = cache[sym]["close"].pct_change()
            clear_regime_cache(sym, timeframe)
    logger.info("Completed OHLCV update for timeframe %s", timeframe)
    return cache


async def update_multi_tf_ohlcv_cache(
    exchange,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    symbols: Iterable[str],
    config: Dict,
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
    priority_queue: Deque[str] | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for multiple timeframes.

    Parameters
    ----------
    config : Dict
        Configuration containing a ``timeframes`` list.
    """
    from crypto_bot.regime.regime_classifier import clear_regime_cache

    limit = max(limit, 200)

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
    logger.info("Updating OHLCV cache for timeframes: %s", tfs)

    min_volume_usd = float(config.get("min_volume_usd", 0) or 0)
    vol_thresh = config.get("bounce_scalper", {}).get("vol_zscore_threshold")

    for tf in tfs:
        logger.info("Starting update for timeframe %s", tf)
        tf_cache = cache.get(tf, {})

        cex_symbols: list[str] = []
        dex_symbols: list[str] = []
        for s in symbols:
            base, _, quote = s.partition("/")
            if quote.upper() == "USDC" and _is_valid_base_token(base):
                dex_symbols.append(s)
            else:
                cex_symbols.append(s)

        if cex_symbols:
            tf_cache = await update_ohlcv_cache(
                exchange,
                tf_cache,
                cex_symbols,
                timeframe=tf,
                limit=limit,
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
                max_concurrent=max_concurrent,
                notifier=notifier,
            )

        for sym in dex_symbols:
            data = None
            vol = 0.0
            try:
                res = await fetch_geckoterminal_ohlcv(
                    sym,
                    timeframe=tf,
                    limit=limit,
                    min_24h_volume=min_volume_usd,
                )
            except Exception as exc:  # pragma: no cover - network
                logger.error("GeckoTerminal OHLCV error for %s: %s", sym, exc)
                res = None

            if res:
                if isinstance(res, tuple):
                    data, vol, *_ = res
                else:
                    data = res
                    vol = min_volume_usd
                add_priority(data, sym)

            if not data or vol < min_volume_usd:
                data = await fetch_dex_ohlcv(
                    exchange,
                    sym,
                    timeframe=tf,
                    limit=limit,
                    min_volume_usd=min_volume_usd,
                    gecko_res=res,
                    use_gecko=False,
                )
                if isinstance(data, Exception) or not data:
                    continue

            if not data:
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
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
    notifier: TelegramNotifier | None = None,
    df_map: Dict[str, Dict[str, pd.DataFrame]] | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for regime detection timeframes."""
    limit = max(limit, 200)
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
            use_websocket=use_websocket,
            force_websocket_history=force_websocket_history,
            max_concurrent=max_concurrent,
            notifier=notifier,
            priority_queue=None,
        )

    return cache

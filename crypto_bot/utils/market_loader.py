"""Utilities for loading trading symbols and fetching OHLCV data."""

from typing import Any, Dict, Optional, List, Iterable, Deque
from dataclasses import dataclass
import asyncio
import inspect
import time
import os
from pathlib import Path
from datetime import datetime, timezone
import yaml
import pandas as pd
import numpy as np
import aiohttp
import base58
import contextlib
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_log,
    before_sleep_log,
)

import ccxt.async_support as ccxt  # type: ignore

try:  # optional redis for caching
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None

from .gecko import gecko_request
from .token_registry import (
    TOKEN_MINTS,
    get_mint_from_gecko,
    fetch_from_helius,
)
from crypto_bot.strategy.evaluator import get_stream_evaluator, StreamEvaluator
from crypto_bot.strategy.registry import load_enabled
from .logger import LOG_DIR, setup_logger
from .constants import NON_SOLANA_BASES
from crypto_bot.data.locks import timeframe_lock, TF_LOCKS as _TF_LOCKS

try:  # optional dependency
    from .telegram import TelegramNotifier
except Exception:  # pragma: no cover - optional
    TelegramNotifier = Any


def utc_now_ms() -> int:
    """Return current UTC time in milliseconds."""
    return int(time.time() * 1000)


def iso_utc(ms: int) -> str:
    """Return an ISO 8601 UTC timestamp for ``ms`` milliseconds."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
 
 
async def get_kraken_listing_date(symbol: str) -> Optional[int]:
    """Return Kraken listing timestamp for *symbol*.

    This is a lightweight placeholder used in tests; production code may
    provide a more complete implementation elsewhere."""

    return None


_last_snapshot_time = 0

logger = setup_logger(__name__, LOG_DIR / "bot.log")

UNSUPPORTED_SYMBOLS: set[str] = {
    "AIBTC/EUR",
    "AIBTC/USD",
}
"""Symbols that consistently fail to load OHLCV data.

Extend this set to skip additional markets without making network
requests.
"""

_UNSUPPORTED_LOGGED: set[str] = set()


def _log_unsupported(symbol: str) -> None:
    if symbol not in _UNSUPPORTED_LOGGED:
        logger.debug("Skipping unsupported symbol %s", symbol)
        _UNSUPPORTED_LOGGED.add(symbol)

# Base suffixes that indicate a synthetic or index pair when appended to
# another asset (e.g. ``AIBTC/USD`` represents the AI/BTC index).
_SYNTH_SUFFIXES = {"BTC", "ETH", "USD", "EUR", "USDT"}


def is_synthetic_symbol(symbol: str) -> bool:
    """Return ``True`` if *symbol* appears to be a synthetic/index pair."""

    base, _, _ = symbol.partition("/")
    base = base.upper()
    return any(base.endswith(sfx) and base != sfx for sfx in _SYNTH_SUFFIXES)


def is_supported_symbol(symbol: str) -> bool:
    """Return ``True`` if *symbol* should be processed for OHLCV."""

    return symbol not in UNSUPPORTED_SYMBOLS and not is_synthetic_symbol(symbol)

async def _maybe_enqueue_eval(symbol: str, timeframe: str, cache: Dict[str, Dict[str, pd.DataFrame]], config: Dict[str, Any]) -> None:
    if timeframe not in ("1m", "5m"):
        return
    try:
        if warmup_reached_for(symbol, timeframe, cache, config):
            logger.info("OHLCV[%s] warmup met for %s \u2192 enqueue for evaluation", timeframe, symbol)
            ctx = {"timeframes": ["1m", "5m"], "symbol": symbol}
            try:
                evaluator = get_stream_evaluator()
            except Exception:
                return
            await evaluator.enqueue(symbol, ctx)
    except Exception:
        pass

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
# Per-timeframe locks are provided by crypto_bot.data.locks

# Shared StreamEvaluator instance set by main
STREAM_EVALUATOR: StreamEvaluator | None = None
_WARMED_UP: set[str] = set()

# Redis settings
REDIS_TTL = 3600  # cache expiry in seconds
_REDIS_URL: str | None = None
_REDIS_CONN: Any | None = None

def _get_redis_conn(url: str | None):
    """Return Redis connection for ``url`` if available."""
    global _REDIS_URL, _REDIS_CONN
    if not url or redis is None or not hasattr(redis, "Redis"):
        return None
    if _REDIS_CONN is None or url != _REDIS_URL:
        try:
            if hasattr(redis.Redis, "from_url"):
                _REDIS_CONN = redis.Redis.from_url(url)
            else:  # pragma: no cover - older redis package
                _REDIS_CONN = redis.Redis()
            _REDIS_URL = url
        except Exception:
            _REDIS_CONN = None
    return _REDIS_CONN


def set_stream_evaluator(ev: StreamEvaluator | None) -> None:
    """Assign global StreamEvaluator used for streaming symbol evaluation."""
    global STREAM_EVALUATOR
    STREAM_EVALUATOR = ev


def warmup_reached_for(
    symbol: str,
    timeframe: str,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    config: Dict,
) -> bool:
    """Return ``True`` if 1m and 5m warmup requirements are met for ``symbol``."""
    warmup_map = config.get("warmup_candles", {}) or {}
    for tf in ("1m", "5m"):
        required = int(warmup_map.get(tf, 1))
        df = cache.get(tf, {}).get(symbol)
        if df is None or len(df) < required:
            return False
    return True

# Mapping of common symbols to CoinGecko IDs for OHLC fallback
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}


# --- NEW: resolve markets only to what the exchange actually lists ---
def resolve_listed_symbol(exchange, base: str, allowed_quotes: list[str]) -> str | None:
    """Return the first listed symbol for *base* across ``allowed_quotes``."""
    markets = getattr(exchange, "markets", {}) or {}
    for q in allowed_quotes:
        sym = f"{base}/{q}"
        if sym in markets:
            return sym
    for m in markets.values():
        try:
            if m.get("base") == base and m.get("quote") in allowed_quotes:
                return m.get("symbol")
        except Exception:
            continue
    return None


# --- NEW: safe closing helpers for ccxt / aiohttp ---
async def _safe_exchange_close(exchange, where: str = ""):
    """Attempt to close ``exchange`` without raising."""
    close = getattr(exchange, "close", None)
    if not close:
        return
    try:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            close()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(f"Exchange.close() failed {where}: {e!r}")


async def fetch_ohlcv_block(exchange_id: str, bases: list[str], timeframe: str, limit: int,
                            allowed_quotes: list[str]):
    """Fetch OHLCV for ``bases`` on ``exchange_id`` over ``timeframe``."""
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    try:
        await ex.load_markets()
        results = {}
        bases_dedup = list(dict.fromkeys(bases))
        for base in bases_dedup:
            symbol = resolve_listed_symbol(ex, base, allowed_quotes)
            if not symbol:
                logger.debug(
                    f"Skipping {base}: no listed market on {exchange_id} for quotes {allowed_quotes}"
                )
                continue
            try:
                candles = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if candles:
                    results[symbol] = candles
                else:
                    logger.debug(f"No candles returned for {symbol} @ {timeframe}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"fetch_ohlcv failed for {symbol} @ {timeframe}: {e!r}"
                )
        return results
    finally:
        await _safe_exchange_close(ex, where=f"{exchange_id}:{timeframe}")

# Cache GeckoTerminal pool addresses and metadata per symbol
# Mapping: symbol -> (pool_addr, volume, reserve, price, limit)
GECKO_POOL_CACHE: dict[str, tuple[str, float, float, float, int]] = {}
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
    priority_symbols: List[str] | None
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
            union_priority: List[str] = []
            for r in reqs:
                union_symbols.extend(r.symbols)
                if r.priority_symbols:
                    union_priority.extend(r.priority_symbols)
            # Deduplicate while preserving order
            seen = set()
            union_symbols = [s for s in union_symbols if not (s in seen or seen.add(s))]
            seen_p = set()
            union_priority = [
                s for s in union_priority if s in union_symbols and not (s in seen_p or seen_p.add(s))
            ]

            base = reqs[0]
            try:
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
                    priority_symbols=union_priority,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.exception(
                    "OHLCV worker: failed on timeframe=%s (batch size=%s). Continuing. Error: %s",
                    base.timeframe,
                    len(union_symbols),
                    e,
                )
                cache = base.cache

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
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    if ohlcv_timeout is None:
        cfg_val = cfg.get("ohlcv_timeout")
        if cfg_val is not None:
            ohlcv_timeout = cfg_val
    if max_failures is None:
        cfg_val = cfg.get("max_ohlcv_failures")
        if cfg_val is not None:
            max_failures = cfg_val
    if max_ws_limit is None:
        cfg_val = cfg.get("max_ws_limit")
        if cfg_val is not None:
            max_ws_limit = cfg_val
    if max_concurrent is None:
        cfg_val = cfg.get("max_concurrent_ohlcv")
        if cfg_val is not None:
            max_concurrent = cfg_val
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


def gecko_timeframe_parts(timeframe: str) -> tuple[str, int]:
    """Return (path, aggregate) for GeckoTerminal OHLCV requests."""
    unit = timeframe[-1]
    value = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 1
    if unit == "m":
        return "minute", value
    if unit == "h":
        return "hour", value
    if unit == "d":
        return "day", value
    return timeframe, 1


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

    # Restrict to commonly traded quotes on Kraken
    allowed_quotes = {"USD", "USDT", "EUR"}
    quotes = df.get("quote", "").astype(str).str.upper()
    mask_quote = quotes.isin(allowed_quotes)
    df.loc[df["reason"].isna() & ~mask_quote, "reason"] = "disallowed_quote"

    df.loc[df["reason"].isna() & df["symbol"].isin(exclude_set), "reason"] = "excluded"

    mask_synth = df["symbol"].apply(is_synthetic_symbol)
    df.loc[df["reason"].isna() & mask_synth, "reason"] = "synthetic"
    synth_count = int((df["reason"] == "synthetic").sum())

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

    logger.info("%d active Kraken pairs discovered", len(symbols))
    if synth_count:
        logger.info("Excluded %d synthetic/index pairs", synth_count)

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
        logger.debug("Skipping %s: OHLCV not supported on %s", symbol, ex_id)
        return []
    if getattr(exchange, "timeframes", None) and timeframe not in getattr(
        exchange, "timeframes", {}
    ):
        ex_id = getattr(exchange, "id", "unknown")
        logger.warning("Timeframe %s not supported on %s", timeframe, ex_id)
        return []

    if not is_supported_symbol(symbol):
        logger.debug("Skipping %s: OHLCV not supported", symbol)
        return []

    if use_websocket or force_websocket_history:
        logger.debug(
            "Websocket flags set for %s but functionality is disabled; using REST",
            symbol,
        )

    if timeframe in ("4h", "1d"):
        use_websocket = False


    try:
        if hasattr(exchange, "markets"):
            markets = getattr(exchange, "markets", {})
            if not markets and hasattr(exchange, "load_markets"):
                try:
                    if asyncio.iscoroutinefunction(
                        getattr(exchange, "load_markets", None)
                    ):
                        markets = await exchange.load_markets()
                    else:
                        markets = await asyncio.to_thread(exchange.load_markets)
                except Exception as exc:
                    logger.warning("load_markets failed: %s", exc)
            if markets and symbol not in markets:
                _log_unsupported(symbol)
                return []
            market_id = markets.get(symbol, {}).get("id", symbol)
        else:
            market_id = symbol

        if limit > 0:
            data_all: list = []
            orig_limit = limit
            while limit > 0:
                req_limit = min(limit, 720)
                params = {"symbol": market_id, "timeframe": timeframe, "limit": req_limit}
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
            if (
                since is not None
                and len(data_all) < orig_limit
                and hasattr(exchange, "fetch_ohlcv")
            ):
                logger.info(
                    "Incomplete OHLCV for %s: got %d of %d",
                    symbol,
                    len(data_all),
                    orig_limit,
                )
                kwargs_r = {"symbol": symbol, "timeframe": timeframe, "limit": orig_limit}
                try:
                    if asyncio.iscoroutinefunction(exchange.fetch_ohlcv):
                        data_r = await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_r,
                        )
                    else:
                        data_r = await _call_with_retry(
                            asyncio.to_thread,
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_r,
                        )
                except asyncio.CancelledError:
                    raise
                if len(data_r) > len(data_all):
                    data_all = data_r
            return data_all
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
                    now_ms = utc_now_ms()
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
                now_ms = utc_now_ms()
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
        logger.error(
            "REST OHLCV timeout for %s on %s (tf=%s limit=%s): %s",
            symbol,
            ex_id,
            timeframe,
            limit,
            exc,
            exc_info=False,
        )
        return []
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network
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

    if not is_supported_symbol(symbol):
        _log_unsupported(symbol)
        return []

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

async def fetch_order_book_async(
    exchange,
    symbol: str,
    depth: int = 2,
) -> dict | Exception:
    """Return order book snapshot for ``symbol`` with top ``depth`` levels."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchOrderBook"):
        return {}

    if not is_supported_symbol(symbol):
        _log_unsupported(symbol)
        return {}

    markets = getattr(exchange, "markets", {})
    if not markets and hasattr(exchange, "load_markets"):
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "load_markets", None)):
                markets = await exchange.load_markets()
            else:
                markets = await asyncio.to_thread(exchange.load_markets)
        except Exception as exc:
            logger.warning("load_markets failed: %s", exc)
    if markets and symbol not in markets:
        _log_unsupported(symbol)
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


async def fetch_ohlcv_from_trades(
    exchange,
    symbol: str,
    timeframe: str,
    since: int | None,
    limit: int,
) -> list:
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
    open_price = high = low = close = float(trades[0][1])
    volume = float(trades[0][2]) if len(trades[0]) > 2 else 0.0

    for t in trades[1:]:
        ts = int(t[0])
        price = float(t[1])
        amount = float(t[2]) if len(t) > 2 else 0.0
        b = ts - ts % tf_ms
        if b != bucket:
            ohlcv.append([bucket, open_price, high, low, close, volume])
            if len(ohlcv) >= limit:
                return ohlcv[:limit]
            bucket = b
            open_price = high = low = close = price
            volume = amount
        else:
            high = max(high, price)
            low = min(low, price)
            close = price
            volume += amount

    ohlcv.append([bucket, open_price, high, low, close, volume])
    return ohlcv[:limit]


async def load_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
    mode: str = "rest",
    **kwargs,
) -> list:
    """Load OHLCV data via REST with basic retries.

    ``mode`` is retained for backward compatibility but ignored. On a
    successful call it sleeps for one second to respect exchange rate limits.
    Any exception containing ``"429"`` triggers a 60 second sleep before the
    request is retried.
    """

    markets = getattr(exchange, "markets", None)
    if markets is not None:
        if not markets and hasattr(exchange, "load_markets"):
            try:
                if asyncio.iscoroutinefunction(exchange.load_markets):
                    markets = await exchange.load_markets()
                else:
                    markets = await asyncio.to_thread(exchange.load_markets)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("load_markets failed: %s", exc)
        if markets and symbol not in markets:
            _log_unsupported(symbol)
            return []
        market_id = markets.get(symbol, {}).get("id", symbol)
    else:
        market_id = symbol

    while True:
        try:
            fetch_fn = getattr(exchange, "fetch_ohlcv")
            if asyncio.iscoroutinefunction(fetch_fn):
                data = await fetch_fn(
                    market_id, timeframe=timeframe, limit=limit, **kwargs
                )
            else:  # pragma: no cover - synchronous fallback
                data = await asyncio.to_thread(
                    fetch_fn, market_id, timeframe, limit, **kwargs
                )
            await asyncio.sleep(1)
            return data
        except Exception as exc:
            if "429" in str(exc):
                await asyncio.sleep(60)
                continue
            await asyncio.sleep(1)
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
    priority_symbols: Iterable[str] | None = None,
) -> Dict[str, list]:
    """Fetch OHLCV data for multiple symbols concurrently.

    Parameters
    ----------
    notifier : TelegramNotifier | None, optional
        If provided, failures will be sent using this notifier.
    """
    if use_websocket or force_websocket_history:
        logger.debug(
            "Websocket parameters ignored in load_ohlcv_parallel; using REST"
        )

    since_map = since_map or {}

    data: Dict[str, list] = {}
    symbols = list(symbols)
    unsupported = [s for s in symbols if not is_supported_symbol(s)]
    for s in unsupported:
        _log_unsupported(s)
        data[s] = []
    symbols = [s for s in symbols if is_supported_symbol(s)]

    markets = getattr(exchange, "markets", None)
    if markets is not None:
        if not markets and hasattr(exchange, "load_markets"):
            try:
                if asyncio.iscoroutinefunction(exchange.load_markets):
                    markets = await exchange.load_markets()
                else:
                    markets = await asyncio.to_thread(exchange.load_markets)
            except Exception as exc:
                logger.warning("load_markets failed: %s", exc)
        missing = [s for s in symbols if s not in markets]
        for s in missing:
            _log_unsupported(s)
            data[s] = []
        symbols = [s for s in symbols if s in markets]

    if not symbols:
        return data

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

    if priority_symbols:
        prio_list: List[str] = []
        seen: set[str] = set()
        for s in priority_symbols:
            if s in symbols and s not in seen:
                prio_list.append(s)
                seen.add(s)
        symbols = prio_list + [s for s in symbols if s not in seen]

    if not symbols:
        return data

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
            kwargs_l = {}
            since_val = since_map.get(sym)
            if since_val is not None:
                kwargs_l["since"] = since_val
            data = await load_ohlcv(
                exchange,
                sym,
                timeframe=timeframe,
                limit=limit,
                mode="rest",
                **kwargs_l,
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

    ex_id = getattr(exchange, "id", "unknown")
    mode = "REST"
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
            status = getattr(res, "http_status", getattr(res, "status", None))
            delay = 60 if status == 429 else RETRY_DELAY
            count = 1
            disabled = False
            if info is not None:
                delay = 60 if status == 429 else min(info["delay"] * 2, MAX_RETRY_DELAY)
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
    priority_symbols: Iterable[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Update cached OHLCV DataFrames with new candles.

    Parameters
    ----------
    max_concurrent : int | None, optional
        Maximum number of concurrent OHLCV requests. ``None`` means no limit.
    start_since : int | None, optional
        When provided, fetch data starting from this timestamp in milliseconds.
    """

    try:  # pragma: no cover - optional regime dependency
        from crypto_bot.regime.regime_classifier import clear_regime_cache
    except Exception:  # pragma: no cover - optional
        clear_regime_cache = lambda *_a, **_k: None

    # Redis warm cache
    redis_conn = _get_redis_conn((config or {}).get("redis_url"))
    if redis_conn:
        for sym in symbols:
            try:
                raw = redis_conn.get(f"ohlcv:{sym}:{timeframe}")
            except Exception:
                raw = None
            if raw:
                try:
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    cache[sym] = pd.read_json(raw, orient="split")
                except Exception:
                    pass

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
        needed = int((utc_now_ms() - start_since) // (tf_sec * 1000)) + 1
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

    t0 = time.time()
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
            use_websocket=use_websocket,
            force_websocket_history=force_websocket_history,
            max_concurrent=max_concurrent,
            notifier=notifier,
            priority_symbols=priority_symbols,
        )
        for sym, rows in batch.items():
            if rows:
                data_map[sym].extend(rows)
                last_ts = rows[-1][0]
                curr_since[sym] = last_ts + timeframe_seconds(exchange, timeframe) * 1000
        if all(len(batch.get(sym, [])) < req_limit for sym in symbols):
            break
        remaining -= req_limit

    count_ok = sum(1 for rows in data_map.values() if rows)
    logger.info(
        "Fetched OHLCV for %d/%d symbols on %s in %.1fs",
        count_ok,
        len(symbols),
        timeframe,
        time.time() - t0,
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
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
                max_concurrent=max_concurrent,
                notifier=notifier,
                priority_symbols=priority_symbols,
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

        if df_new is None or df_new.empty or "timestamp" not in df_new.columns:
            logger.info(
                "OHLCV: no new data for %s @ %s; keeping existing cache", sym, timeframe
            )
            continue

        df_new["timestamp"] = pd.to_numeric(df_new["timestamp"], errors="coerce")
        df_new = df_new.dropna(subset=["timestamp"])
        if df_new.empty:
            logger.info(
                "OHLCV: non-numeric/empty timestamps for %s @ %s; skipping update",
                sym,
                timeframe,
            )
            continue

        unit = "ms" if df_new["timestamp"].max() > 1e12 else "s"
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit=unit, utc=True)
        df_new = df_new.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
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
                use_websocket=False,
                force_websocket_history=force_websocket_history,
                max_concurrent=max_concurrent,
                notifier=notifier,
                priority_symbols=priority_symbols,
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
            if redis_conn:
                try:
                    redis_conn.setex(
                        f"ohlcv:{sym}:{timeframe}",
                        REDIS_TTL,
                        cache[sym].to_json(orient="split"),
                    )
                except Exception:
                    pass
    logger.info("Completed OHLCV update for timeframe %s", timeframe)
    return cache


# --- Additional helpers ----------------------------------------------------


async def fetch_coingecko_ohlc(symbol: str, days: int = 1) -> List[List[float]] | None:
    """Return OHLC data from CoinGecko for ``symbol``.

    The returned rows follow the CCXT OHLCV format with volume set to ``0`` as
    the CoinGecko endpoint does not provide volume information.
    """

    base, _, quote = symbol.partition("/")
    coin_id = COINGECKO_IDS.get(base.upper())
    if not coin_id:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": quote.lower(), "days": days}
    try:
        data = await gecko_request(url, params=params)
        return [
            [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), 0.0]
            for row in data
        ]
    except Exception:
        return None


async def fetch_dex_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    min_volume_usd: float | int = 0,
    gecko_res: Any | None = None,
    use_gecko: bool = True,
) -> List[List[float]] | None:
    """Fetch OHLCV data for DEX tokens with several fallbacks."""

    if min_volume_usd:
        logger.debug(
            "min_volume_usd parameter is deprecated and ignored (value=%s)",
            min_volume_usd,
        )

    if gecko_res:
        return gecko_res[0] if isinstance(gecko_res, tuple) else gecko_res

    base, _, quote = symbol.partition("/")
    quote = quote.upper()

    # Try CoinGecko for known USD-quoted pairs
    if use_gecko and quote in SUPPORTED_USD_QUOTES:
        data = fetch_coingecko_ohlc(symbol)
        if inspect.isawaitable(data):
            data = await data
        if data:
            return data

        # Fallback to Coinbase if available
        try:
            if hasattr(ccxt, "coinbase"):
                cb = ccxt.coinbase()
                return await load_ohlcv(cb, symbol, timeframe=timeframe, limit=limit)
        except Exception:
            pass

    # Final fallback: use the provided exchange
    try:
        return await load_ohlcv(
            exchange, symbol, timeframe=timeframe, limit=limit
        )
    except Exception:
        return None


# --- Back-compat: GeckoTerminal OHLCV wrapper using CCXT (real fetch, no stubs) ---
async def fetch_geckoterminal_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    since: Optional[int] = None,
    limit: int = 500,
    exchange: Any = None,
) -> List[List[float]]:
    """Fetch OHLCV using CCXT, resolving symbols to listed markets."""
    if exchange is not None and hasattr(exchange, "fetch_ohlcv"):
        fetch_fn = getattr(exchange, "fetch_ohlcv")
        if asyncio.iscoroutinefunction(fetch_fn):
            return await fetch_fn(symbol, timeframe=timeframe, since=since, limit=limit)
        return await asyncio.to_thread(
            fetch_fn, symbol, timeframe, since, limit
        )

    ex_name = os.environ.get("EXCHANGE", "kraken").lower()
    ex_cls = getattr(ccxt, ex_name, None) or getattr(ccxt, "kraken")
    ex = ex_cls({"enableRateLimit": True})
    try:
        await ex.load_markets()
        base, _, quote = symbol.partition("/")
        allowed = [quote, "USD", "USDT", "EUR"]
        allowed = [q for i, q in enumerate(allowed) if q and q not in allowed[:i]]
        resolved = resolve_listed_symbol(ex, base, allowed)
        if not resolved:
            logger.debug(
                f"Skipping {base}: no listed market on {ex_name} for quotes {allowed}"
            )
            return []
        symbol = resolved
        if asyncio.iscoroutinefunction(ex.fetch_ohlcv):
            return await ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        return await asyncio.to_thread(
            ex.fetch_ohlcv, symbol, timeframe, since, limit
        )
    finally:
        await _safe_exchange_close(ex, where=f"{ex_name}:{timeframe}")


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
    priority_symbols: Iterable[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Batch OHLCV updates for multiple calls."""

    config = config or {}
    backfill_map = config.get("backfill_days", {}) or {}
    warmup_map = config.get("warmup_candles", {}) or {}
    now_ms = utc_now_ms()
    if start_since is not None:
        bf_days = backfill_map.get(timeframe)
        if bf_days is not None:
            cutoff = now_ms - int(float(bf_days) * 86400000)
            if start_since < cutoff:
                logger.info(
                    "Clamping backfill for %s to %d days (%s)",
                    timeframe,
                    bf_days,
                    iso_utc(cutoff),
                )
                start_since = cutoff
    warmup = warmup_map.get(timeframe)
    if warmup is not None and limit > int(warmup):
        logger.info(
            "Clamping warmup candles for %s to %d (was %d)",
            timeframe,
            warmup,
            limit,
        )
        limit = int(warmup)
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
        list(priority_symbols) if priority_symbols else None,
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
    try:  # pragma: no cover - optional regime dependency
        from crypto_bot.regime.regime_classifier import clear_regime_cache
    except Exception:  # pragma: no cover - optional
        clear_regime_cache = lambda *_a, **_k: None

    limit = int(limit)
    # Use the limit provided by the caller

    # Ensure warmup candles satisfy strategy indicator lookbacks. This will
    # either raise the configured warmup or disable strategies requiring more
    # history, depending on ``data.auto_raise_warmup``.
    load_enabled(config)

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

    symbols = list(symbols)
    priority_syms: list[str] = []
    if priority_queue is not None:
        seen: set[str] = set()
        while priority_queue:
            sym = priority_queue.popleft()
            if sym in symbols and sym not in seen:
                priority_syms.append(sym)
                seen.add(sym)

    for tf in tfs:
        lock = timeframe_lock(tf)
        if lock.locked():
            logger.info("Skip: %s update already running.", tf)
            continue
        async with lock:
            logger.info("Starting OHLCV update for timeframe %s", tf)
            tf_cache = cache.get(tf, {})

            now_ms = utc_now_ms()
            tf_sec = timeframe_seconds(exchange, tf)
            dynamic_limits: dict[str, int] = {}
            snapshot_cap = int(config.get("ohlcv_snapshot_limit", limit))
            max_cap = min(snapshot_cap, 720)

            backfill_map = config.get("backfill_days", {}) or {}
            warmup_map = config.get("warmup_candles", {}) or {}
            tf_start = start_since
            bf_days = backfill_map.get(tf)
            if tf_start is not None and bf_days is not None:
                cutoff = now_ms - int(float(bf_days) * 86400000)
                if tf_start < cutoff:
                    logger.info(
                        "Clamping backfill for %s to %d days (%s)",
                        tf,
                        bf_days,
                        iso_utc(cutoff),
                    )
                    tf_start = cutoff
            tf_limit = int(limit)
            needed: int | None = None
            if tf_start is not None:
                needed = int((now_ms - tf_start) // (tf_sec * 1000)) + 1
                tf_limit = max(tf_limit, needed)
            warmup = warmup_map.get(tf)
            if warmup is not None and tf_limit > int(warmup):
                logger.info(
                    "Clamping warmup candles for %s to %d (was %d)",
                    tf,
                    warmup,
                    tf_limit,
                )
                tf_limit = int(warmup)
            tf_start_since = tf_start
            if needed is not None and tf_limit < needed:
                logger.info(
                    "Warmup limit %d smaller than requested range %d for %s; dropping start_since",
                    tf_limit,
                    needed,
                    tf,
                )
                tf_start_since = None

            concurrency = int(config.get("listing_date_concurrency", 5) or 0)
            semaphore = asyncio.Semaphore(concurrency) if concurrency > 0 else None

            async def _fetch_listing(sym: str) -> tuple[str, int | None]:
                if semaphore is not None:
                    async with semaphore:
                        ts = await get_kraken_listing_date(sym)
                else:
                    ts = await get_kraken_listing_date(sym)
                return sym, ts

            start_list = time.perf_counter()
            tasks = [asyncio.create_task(_fetch_listing(sym)) for sym in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(symbols, results):
                if isinstance(res, Exception):
                    logger.exception(
                        "OHLCV task failed for %s @ %s: %s",
                        sym,
                        tf,
                        res,
                    )
                    continue
                _, listing_ts = res
                if listing_ts and 0 < listing_ts <= now_ms:
                    age_ms = now_ms - listing_ts
                    tf_sec = timeframe_seconds(exchange, tf)
                    hist_candles = age_ms // (tf_sec * 1000)
                    if hist_candles <= 0:
                        continue
                    if hist_candles > snapshot_cap * 1000:
                        logger.info(
                            "Skipping OHLCV history for %s on %s (age %d candles)",
                            sym,
                            tf,
                            hist_candles,
                        )
                        continue
                    dynamic_limits[sym] = int(min(hist_candles, max_cap))
            logger.debug(
                "listing date fetch for %d symbols took %.2fs",
                len(symbols),
                time.perf_counter() - start_list,
            )

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

            if priority_syms:
                prio_set = set(priority_syms)
                cex_symbols = [s for s in priority_syms if s in cex_symbols] + [s for s in cex_symbols if s not in prio_set]
                dex_symbols = [s for s in priority_syms if s in dex_symbols] + [s for s in dex_symbols if s not in prio_set]

            if cex_symbols:
                if tf_start_since is None:
                    groups: Dict[int, list[str]] = {}
                    for sym in cex_symbols:
                        sym_limit = dynamic_limits.get(sym, tf_limit)
                        groups.setdefault(int(sym_limit), []).append(sym)
                    for lim, syms in groups.items():
                        curr_limit = tf_limit
                        if lim < tf_limit:
                            for s in syms:
                                logger.info(
                                    "Adjusting limit for %s on %s to %d", s, tf, lim
                                )
                            curr_limit = lim
                        tf_cache = await update_ohlcv_cache(
                            exchange,
                            tf_cache,
                            syms,
                            timeframe=tf,
                            limit=curr_limit,
                            config={
                                "min_history_fraction": 0,
                                "ohlcv_batch_size": config.get("ohlcv_batch_size"),
                            },
                            batch_size=batch_size,
                            start_since=tf_start_since,
                            use_websocket=use_websocket,
                            force_websocket_history=force_websocket_history,
                            max_concurrent=max_concurrent,
                            notifier=notifier,
                            priority_symbols=priority_syms,
                        )
                else:
                    from crypto_bot.main import update_df_cache

                    for sym in cex_symbols:
                        batches: list = []
                        current_since = tf_start_since
                        sym_total = min(tf_limit, dynamic_limits.get(sym, tf_limit))
                        if sym_total < tf_limit:
                            logger.info(
                                "Adjusting limit for %s on %s to %d", sym, tf, sym_total
                            )
                        remaining = sym_total
                        while remaining > 0:
                            req = min(remaining, 1000)
                            data = await load_ohlcv(
                                exchange,
                                sym,
                                timeframe=tf,
                                limit=req,
                                mode="rest",
                                since=current_since,
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
                            logger.info(
                                "OHLCV: empty or missing 'timestamp' for %s @ %s; skipping update.",
                                sym,
                                tf,
                            )
                            continue

                        df_new = pd.DataFrame(
                            batches,
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        tf_sec = timeframe_seconds(None, tf)

                        # Guard for empty/malformed OHLCV responses.
                        if df_new is None or df_new.empty or "timestamp" not in df_new.columns:
                            logger.info(
                                "OHLCV: empty or missing 'timestamp' for %s @ %s; skipping update.",
                                sym,
                                tf,
                            )
                            continue

                        # Coerce and validate the first timestamp safely (avoid out-of-bounds on iloc[0]).
                        try:
                            ts0 = pd.to_numeric(df_new["timestamp"].iloc[0], errors="coerce")
                        except Exception:
                            ts0 = None

                        if ts0 is None or pd.isna(ts0):
                            logger.info(
                                "OHLCV: first timestamp is NaN/invalid for %s @ %s; skipping update.",
                                sym,
                                tf,
                            )
                            continue

                        # Robust unit detection:
                        if ts0 > 1e14:
                            unit = "ns"
                        elif ts0 > 1e12:
                            unit = "us"
                        elif ts0 > 1e10:
                            unit = "ms"
                        else:
                            unit = "s"

                        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit=unit)
                        df_new = (
                            df_new.set_index("timestamp")
                            .resample(f"{tf_sec}s")
                            .agg(
                                {
                                    "open": "first",
                                    "high": "max",
                                    "low": "min",
                                    "close": "last",
                                    "volume": "sum",
                                }
                            )
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
                        if (
                            STREAM_EVALUATOR
                            and tf in ("1m", "5m")
                            and warmup_reached_for(sym, tf, cache, config)
                        ):
                            if sym not in _WARMED_UP:
                                logger.info(
                                    "OHLCV[%s] warmup met for %s → enqueue for evaluation",
                                    tf,
                                    sym,
                                )
                                _WARMED_UP.add(sym)
                            await STREAM_EVALUATOR.enqueue(
                                sym, {"df_cache": cache, "symbol": sym}
                            )
                        await _maybe_enqueue_eval(sym, tf, cache, config)

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
                        try:
                            res = fetch_geckoterminal_ohlcv(
                                sym,
                                timeframe=tf,
                                limit=sym_l,
                                min_24h_volume=min_volume_usd,
                            )
                        except TypeError:
                            res = fetch_geckoterminal_ohlcv(
                                sym,
                                timeframe=tf,
                                limit=sym_l,
                            )
                        if inspect.isawaitable(res):
                            res = await res
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
                        gecko_res=None,
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
                    if (
                        STREAM_EVALUATOR
                        and tf in ("1m", "5m")
                        and warmup_reached_for(sym, tf, cache, config)
                    ):
                        if sym not in _WARMED_UP:
                            logger.info(
                                "OHLCV[%s] warmup met for %s → enqueue for evaluation",
                                tf,
                                sym,
                            )
                            _WARMED_UP.add(sym)
                        await STREAM_EVALUATOR.enqueue(
                            sym, {"df_cache": cache, "symbol": sym}
                        )

                    cache[tf] = tf_cache
                    await _maybe_enqueue_eval(sym, tf, cache, config)
            cache[tf] = tf_cache
            logger.info("Completed OHLCV update for timeframe %s", tf)

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

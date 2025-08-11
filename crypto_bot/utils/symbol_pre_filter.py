"""Utility for filtering trading pairs based on liquidity and volatility."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Dict

try:  # pragma: no cover - optional dependency
    import ccxt.pro as ccxt  # type: ignore
except Exception:  # pragma: no cover - fall back to standard ccxt
    import ccxt  # type: ignore
try:  # pragma: no cover - import fallback
    from ccxt.base.errors import NetworkError as CCXTNetworkError
except Exception:  # pragma: no cover - import fallback
    CCXTNetworkError = getattr(ccxt, "NetworkError", Exception)

import aiohttp
import numpy as np
import pandas as pd
import yaml
from cachetools import TTLCache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_log,
    before_sleep_log,
    retry_if_exception,
)
import logging

from .logger import LOG_DIR, setup_logger
from .market_loader import (
    fetch_ohlcv_async,
    update_ohlcv_cache,
    update_multi_tf_ohlcv_cache,
    fetch_geckoterminal_ohlcv,
    timeframe_seconds,
)
from .constants import NON_SOLANA_BASES
from .correlation import incremental_correlation
from .symbol_scoring import score_symbol
from .telemetry import telemetry
from .pair_cache import PAIR_FILE, load_liquid_map

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    cfg = {}

# Semaphore guarding concurrent OHLCV requests
SEMA: asyncio.Semaphore | None = None
# Semaphore guarding concurrent ticker requests
TICKER_SEMA: asyncio.Semaphore | None = None
TICKER_DELAY = 0.0


def init_semaphore(limit: int | None = None) -> asyncio.Semaphore:
    """Initialize the global semaphore with ``limit`` permits."""

    global SEMA
    if limit is None:
        limit = 4
    try:
        val = int(limit)
        if val < 1:
            raise ValueError
    except (TypeError, ValueError):
        val = 4
    SEMA = asyncio.Semaphore(val)
    return SEMA


def init_ticker_semaphore(
    limit: int | None = None, delay_ms: int | float | None = None
) -> asyncio.Semaphore:
    """Initialize the ticker semaphore and delay."""

    global TICKER_SEMA, TICKER_DELAY
    if limit is None:
        limit = 20
    try:
        val = int(limit)
        if val < 1:
            raise ValueError
    except (TypeError, ValueError):
        val = 20
    TICKER_SEMA = asyncio.Semaphore(val)
    try:
        delay_val = float(delay_ms) if delay_ms is not None else 0.0
    except (TypeError, ValueError):
        delay_val = 0.0
    TICKER_DELAY = delay_val / 1000 if delay_val else 0.0
    return TICKER_SEMA


logger = setup_logger(__name__, LOG_DIR / "symbol_filter.log")

API_URL = "https://api.kraken.com/0/public"
DEFAULT_MIN_VOLUME_USD = 1000  # Lower from 50000 for testing/Solana volatility
DEFAULT_VOLUME_PERCENTILE = 10  # Lower from 30 to include more pairs
DEFAULT_CHANGE_PCT_PERCENTILE = 50

# Mapping of exchange specific symbols to standardized forms
_ALIASES = {"XBT": "BTC", "XBTUSDT": "BTC/USDT"}


def _norm_symbol(sym: str) -> str:
    """Return ``sym`` normalized for consistent lookups."""

    sym = sym.upper().replace("XBT", "BTC")
    if sym in _ALIASES:
        return _ALIASES[sym]
    if "/" not in sym and sym.endswith("USDT"):
        return sym[:-4] + "/USDT"
    return sym


# cache for ticker data when using watchTickers
ticker_cache: dict[str, dict] = {}
ticker_ts: dict[str, float] = {}
# track ticker fetch failures per symbol
ticker_failures: dict[str, dict] = {}

# default backoff settings for ticker retries
TICKER_BACKOFF_INITIAL = 2
TICKER_BACKOFF_MAX = 60

# Cache of recent liquidity metrics per symbol
liq_cache = TTLCache(maxsize=2000, ttl=900)


# tenacity wrapped helper for WebSocket ticker requests
def _ws_retry_filter(exc: Exception) -> bool:
    """Return ``True`` to retry unless a WebSocket closed with code 1006."""

    return not (
        isinstance(exc, CCXTNetworkError) and getattr(exc, "code", None) == 1006
    )


@retry(
    wait=wait_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
    before=before_log(logger, logging.DEBUG),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_exception(_ws_retry_filter),
)
async def _watch_tickers_with_retry(exchange, symbols):
    """Call ``exchange.watch_tickers`` with retries."""

    try:
        return await exchange.watch_tickers(symbols)
    except Exception:
        await asyncio.sleep(1)
        raise


async def has_enough_history(
    exchange, symbol: str, days: int = 30, timeframe: str = "1d", min_fraction: float = 0.5
) -> bool:
    """Return ``True`` when ``symbol`` has at least ``days`` days of history.

    ``min_fraction`` specifies how much of ``days`` worth of candles are required
    for a symbol to pass.  This helps include recently listed tokens that have
    partial history available.
    """
    seconds = timeframe_seconds(exchange, timeframe)
    candles_needed = int((days * 86400) / seconds)
    try:
        async with SEMA:
            data = await fetch_ohlcv_async(
                exchange, symbol, timeframe=timeframe, limit=candles_needed
            )
    except Exception as exc:  # pragma: no cover - network
        logger.warning(
            "fetch_ohlcv failed for %s on %s for %d days: %s",
            symbol,
            timeframe,
            days,
            exc,
            exc_info=True,
        )
        return False
    if isinstance(data, Exception):  # pragma: no cover - network
        logger.warning(
            "fetch_ohlcv returned exception for %s on %s for %d days: %s",
            symbol,
            timeframe,
            days,
            data,
        )
        return False
    if not isinstance(data, list) or len(data) < candles_needed * min_fraction:
        return False
    return True


async def _fetch_ticker_async(
    pairs: Iterable[str], timeout: int = 10, exchange=None
) -> dict:
    """Return ticker data for ``pairs`` in batches of 20 using aiohttp.

    If ``exchange`` is provided each input symbol is converted to the
    exchange specific market id (via ``exchange.market_id``) before
    sending the request.  This ensures, for example, that Kraken receives
    ``XBTUSDT`` instead of ``BTCUSDT``.
    """

    mock = os.getenv("MOCK_KRAKEN_TICKER")
    if mock:
        return json.loads(mock)

    pairs_list = []
    for sym in pairs:
        if exchange is not None and hasattr(exchange, "market_id"):
            try:
                market_id = exchange.market_id(sym)
            except Exception:
                market_id = sym
        else:
            market_id = sym
        pairs_list.append(market_id.replace("/", ""))
    cfg_local = globals().get("cfg", {})
    rate_ms = cfg_local.get("ticker_rate_limit")
    if rate_ms is None:
        rate_ms = getattr(exchange, "rateLimit", 0) if exchange is not None else 0
    delay = rate_ms / 1000 if rate_ms else 0
    combined: dict = {"result": {}, "error": []}
    async with aiohttp.ClientSession() as session:
        async def fetch(url: str):
            async with TICKER_SEMA:
                resp = await session.get(url, timeout=timeout)
                if delay:
                    await asyncio.sleep(delay)
                return resp

        tasks = []
        for i in range(0, len(pairs_list), 20):
            chunk = pairs_list[i : i + 20]
            url = f"{API_URL}/Ticker?pair={','.join(chunk)}"
            tasks.append(asyncio.create_task(fetch(url)))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            resp.raise_for_status()
            data = await resp.json()
            combined["error"] += data.get("error", [])
            combined["result"].update(data.get("result", {}))

    return combined


def _id_for_symbol(exchange, symbol: str) -> str:
    """Return Kraken pair id for ``symbol`` using ``exchange.market_id`` when available."""
    try:
        market_id = getattr(exchange, "market_id", None)
        if callable(market_id):
            return market_id(symbol)
    except Exception:  # pragma: no cover - best effort
        pass
    return symbol.replace("/", "")


USD_STABLES = {"USD", "USDT", "USDC"}


async def _parse_metrics(
    exchange, symbol: str, ticker: dict
) -> tuple[float, float, float]:
    """Return volume USD, percent change and spread percentage using cache."""

    if "c" in ticker and isinstance(ticker["c"], (list, tuple)):
        # Raw Kraken format
        last = float(ticker["c"][0])
        open_price = float(ticker.get("o", last))
        ask = float(ticker.get("a", [0])[0])
        bid = float(ticker.get("b", [0])[0])
        vwap = float(ticker.get("p", [last, last])[1])
        volume = float(ticker.get("v", [0, 0])[1])
        volume_usd = volume * vwap
    else:
        # CCXT normalized ticker
        last = float(ticker.get("last") or ticker.get("close") or 0.0)
        open_price = float(ticker.get("open", last))
        ask = float(ticker.get("ask", 0.0))
        bid = float(ticker.get("bid", 0.0))
        vwap = float(ticker.get("vwap", last))
        if ticker.get("quoteVolume") is not None:
            volume_usd = float(ticker.get("quoteVolume"))
        else:
            base_vol = float(ticker.get("baseVolume", 0.0))
            volume_usd = base_vol * vwap

    base, _, quote = symbol.partition("/")
    quote = quote.upper()
    if quote and quote not in USD_STABLES:
        price = None
        pair = ""
        for cur in USD_STABLES:
            pair = f"{quote}/{cur}"
            cached = ticker_cache.get(pair)
            if cached:
                if "c" in cached and isinstance(cached["c"], (list, tuple)):
                    price = float(cached["c"][0])
                else:
                    price = float(cached.get("last") or cached.get("close") or 0.0)
                break
        if price is None and hasattr(exchange, "fetch_ticker"):
            fetch = exchange.fetch_ticker
            for cur in USD_STABLES:
                pair = f"{quote}/{cur}"
                try:
                    if asyncio.iscoroutinefunction(fetch):
                        t = await fetch(pair)
                    else:
                        t = await asyncio.to_thread(fetch, pair)
                    if "c" in t and isinstance(t["c"], (list, tuple)):
                        price = float(t["c"][0])
                    else:
                        price = float(t.get("last") or t.get("close") or 0.0)
                    ticker_cache[pair] = t
                    break
                except Exception:
                    continue
        if price:
            volume_usd *= price

    change_pct = ((last - open_price) / open_price) * 100 if open_price else 0.0

    spread_pct = abs(ask - bid) / last * 100 if last else 0.0
    liq_cache[symbol] = (volume_usd, spread_pct, 1.0)

    cached_vol, cached_spread, _ = liq_cache[symbol]
    logger.info("%s: vol=%.2f chg=%.2f%% spr=%.2f%%", symbol, cached_vol, change_pct, cached_spread)
    return cached_vol, change_pct, cached_spread


async def _refresh_tickers(
    exchange,
    symbols: Iterable[str],
    config: dict | None = None,
) -> dict:
    """Return ticker data using WS/HTTP and fall back to per-symbol fetch."""

    start_time = time.perf_counter()

    symbols = list(symbols)
    cfg = config if config is not None else globals().get("cfg", {})
    sf = cfg.get("symbol_filter", {})
    init_ticker_semaphore(
        sf.get("max_concurrent_tickers", cfg.get("max_concurrent_tickers")),
        cfg.get("ticker_rate_limit"),
    )
    delay_ms = cfg.get("ticker_rate_limit")
    if delay_ms is None:
        delay_ms = getattr(exchange, "rateLimit", 0)
    delay = delay_ms / 1000 if delay_ms else 0
    attempts = int(sf.get("ticker_retry_attempts", 3))
    if attempts < 1:
        attempts = 1
    log_exc = sf.get("log_ticker_exceptions", False)

    backoff_initial = float(cfg.get("ticker_backoff_initial", TICKER_BACKOFF_INITIAL))
    backoff_max = float(cfg.get("ticker_backoff_max", TICKER_BACKOFF_MAX))

    now = time.time()
    batch = cfg.get("symbol_filter", {}).get("kraken_batch_size", 100)
    timeout = cfg.get("symbol_filter", {}).get("http_timeout", 10)
    symbols = list(symbols)
    filtered: list[str] = []
    for s in symbols:
        info = ticker_failures.get(s)
        if info and now - info["time"] < info["delay"]:
            continue
        filtered.append(s)
    symbols = filtered
    if not hasattr(exchange, "options"):
        exchange.options = {}
    markets = getattr(exchange, "markets", None)
    if markets is not None:
        if not markets and hasattr(exchange, "load_markets"):
            try:
                if asyncio.iscoroutinefunction(exchange.load_markets):
                    await exchange.load_markets()
                else:
                    await asyncio.to_thread(exchange.load_markets)
                markets = getattr(exchange, "markets", markets)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("load_markets failed: %s", exc)
        missing = [s for s in symbols if s not in markets]
        if missing:
            logger.warning(
                "Skipping symbols not in exchange.markets: %s",
                ", ".join(missing),
            )
        symbols = [s for s in symbols if s in markets]

    attempted_syms: set[str] = set()

    def record_results(res: dict):
        for sym in attempted_syms:
            if sym in res and res[sym]:
                ticker_failures.pop(sym, None)
            else:
                info = ticker_failures.get(sym)
                delay = backoff_initial
                count = 1
                if info is not None:
                    delay = min(info["delay"] * 2, backoff_max)
                    count = info.get("count", 0) + 1
                ticker_failures[sym] = {"time": now, "delay": delay, "count": count}

    try_ws = (
        getattr(getattr(exchange, "has", {}), "get", lambda _k: False)("watchTickers")
        and getattr(exchange, "options", {}).get("ws_scan", True)
    )
    ws_failures = int(getattr(exchange, "options", {}).get("ws_failures", 0))
    ws_limit = int(cfg.get("ws_failures_before_disable", 3))
    try_http = True
    data: dict = {}

    if try_ws:
        to_fetch = [
            s for s in symbols if now - ticker_ts.get(s, 0) > 5 or s not in ticker_cache
        ]
        if to_fetch:
            attempted_syms.update(to_fetch)
            if ws_failures:
                await asyncio.sleep(min(2 ** (ws_failures - 1), 30))
            try:
                async with TICKER_SEMA:
                    data = await _watch_tickers_with_retry(exchange, to_fetch)
                    if delay:
                        await asyncio.sleep(delay)
                opts = getattr(exchange, "options", None)
                if opts is not None:
                    opts["ws_failures"] = 0
            except Exception as exc:  # pragma: no cover - network
                if isinstance(exc, CCXTNetworkError) and getattr(exc, "code", None) == 1006:
                    logger.warning(
                        "WebSocket closed (1006) \u2013 falling back to HTTP",
                    )
                    telemetry.inc("scan.api_errors")
                    opts = getattr(exchange, "options", None)
                    if opts is not None:
                        failures = opts.get("ws_failures", 0) + 1
                        opts["ws_failures"] = failures
                        opts["ws_scan"] = False
                    telemetry.inc("scan.ws_errors")
                else:
                    logger.warning(
                        "watch_tickers failed: %s \u2013 falling back to HTTP. Consider setting exchange.options.ws_scan to False if this continues.",
                        exc,
                        exc_info=log_exc,
                    )
                    telemetry.inc("scan.api_errors")
                    opts = getattr(exchange, "options", None)
                    if opts is not None:
                        failures = opts.get("ws_failures", 0) + 1
                        opts["ws_failures"] = failures
                        if failures >= ws_limit:
                            if opts.get("ws_scan", True):
                                logger.warning(
                                    "Disabling WebSocket scanning after %d errors",
                                    failures,
                                )
                            opts["ws_scan"] = False
                    telemetry.inc("scan.ws_errors")
                attempted_syms.clear()
                try_ws = False
                try_http = True
        for sym, ticker in data.items():
            if sym not in symbols:
                continue
            ticker_cache[sym] = ticker.get("info", ticker)
            ticker_ts[sym] = now
        result = {s: ticker_cache[s] for s in symbols if s in ticker_cache}
        if result:
            record_results(result)
            return {s: t.get("info", t) for s, t in result.items()}

    if try_http:
        if getattr(getattr(exchange, "has", {}), "get", lambda _k: False)(
            "fetchTickers"
        ):
            data = {}
            attempted_syms.update(symbols)
            symbols_list = list(symbols)
            for i in range(0, len(symbols_list), batch):
                chunk = symbols_list[i : i + batch]
                chunk_data: dict | None = None
                for attempt in range(3):
                    try:
                        fetcher = getattr(exchange, "fetch_tickers", None)
                        if asyncio.iscoroutinefunction(fetcher):
                            async with TICKER_SEMA:
                                fetched = await fetcher(chunk)
                                if delay:
                                    await asyncio.sleep(delay)
                        else:
                            async with TICKER_SEMA:
                                fetched = await asyncio.to_thread(fetcher, chunk)
                                if delay:
                                    await asyncio.sleep(delay)
                        chunk_data = {s: t.get("info", t) for s, t in fetched.items()}
                        break
                    except ccxt.BadSymbol as exc:  # pragma: no cover - network
                        logger.warning(
                            "fetch_tickers BadSymbol for %s: %s",
                            ", ".join(chunk),
                            exc,
                        )
                        telemetry.inc("scan.api_errors")
                        return {}
                    except (ccxt.ExchangeError, CCXTNetworkError) as exc:  # pragma: no cover - network
                        if getattr(exc, "http_status", None) in (520, 522) and attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        logger.warning(
                            "fetch_tickers failed: %s \u2013 falling back to Kraken REST /Ticker",
                            exc,
                            exc_info=True,
                        )
                        telemetry.inc("scan.api_errors")
                        chunk_data = None
                        break
                    except Exception as exc:  # pragma: no cover - network
                        logger.warning(
                            "fetch_tickers failed: %s \u2013 falling back to Kraken REST /Ticker",
                            exc,
                            exc_info=True,
                        )
                        telemetry.inc("scan.api_errors")
                        chunk_data = None
                        break
                if chunk_data is None:
                    for attempt in range(attempts):
                        try:
                            fetcher = getattr(exchange, "fetch_tickers", None)
                            if asyncio.iscoroutinefunction(fetcher):
                                async with TICKER_SEMA:
                                    fetched = await fetcher(list(symbols))
                                    if delay:
                                        await asyncio.sleep(delay)
                            else:
                                async with TICKER_SEMA:
                                    fetched = await asyncio.to_thread(fetcher, list(symbols))
                                    if delay:
                                        await asyncio.sleep(delay)
                            data = {s: t.get("info", t) for s, t in fetched.items()}
                            break
                        except ccxt.BadSymbol as exc:  # pragma: no cover - network
                            logger.warning(
                                "fetch_tickers BadSymbol for %s: %s",
                                ", ".join(symbols),
                                exc,
                            )
                            telemetry.inc("scan.api_errors")
                            return {}
                        except (
                            ccxt.ExchangeError,
                            CCXTNetworkError,
                        ) as exc:  # pragma: no cover - network
                            if getattr(exc, "http_status", None) in (520, 522) and attempt < 2:
                                await asyncio.sleep(2**attempt)
                        except (ccxt.ExchangeError, CCXTNetworkError) as exc:  # pragma: no cover - network
                            if getattr(exc, "http_status", None) in (520, 522) and attempt < attempts - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            logger.warning(
                                "fetch_tickers failed: %s \u2013 falling back to Kraken REST /Ticker",
                                exc,
                                exc_info=log_exc,
                            )
                            telemetry.inc("scan.api_errors")
                            data = {}
                            break
                        except Exception as exc:  # pragma: no cover - network
                            logger.warning(
                                "fetch_tickers failed: %s \u2013 falling back to Kraken REST /Ticker",
                                exc,
                                exc_info=log_exc,
                            )
                            telemetry.inc("scan.api_errors")
                            data = {}
                            break
                    else:
                        data = {}
                else:
                    data.update(chunk_data)

            if not data:
                try:
                    pairs = [_id_for_symbol(exchange, s) for s in symbols]

                    try:
                        async with TICKER_SEMA:
                            resp = await _fetch_ticker_async(pairs, timeout=timeout)
                            if delay:
                                await asyncio.sleep(delay)
                    except TypeError:
                        async with TICKER_SEMA:
                            resp = await _fetch_ticker_async(pairs)
                            if delay:
                                await asyncio.sleep(delay)
                    raw = resp.get("result", {})
                    if resp.get("error"):
                        logger.warning(
                            "Ticker API errors for %s: %s",
                            ", ".join(pairs),
                            "; ".join(resp["error"]),
                        )
                        try:
                            async with TICKER_SEMA:
                                raw = (
                                    await _fetch_ticker_async(symbols, timeout=timeout, exchange=exchange)
                                ).get("result", {})
                                if delay:
                                    await asyncio.sleep(delay)
                        except TypeError:
                            async with TICKER_SEMA:
                                raw = (
                                    await _fetch_ticker_async(symbols, exchange=exchange)
                                ).get("result", {})
                                if delay:
                                    await asyncio.sleep(delay)
                    data = {}
                    if len(raw) == len(symbols):
                        for sym, (_, ticker) in zip(symbols, raw.items()):
                            if ticker:
                                data[sym] = ticker
                            else:
                                logger.warning("Empty ticker result for %s", sym)
                    else:
                        extra = iter(raw.values())
                        for sym, pair in zip(symbols, pairs):
                            ticker = raw.get(pair) or raw.get(pair.upper())
                            if ticker is None:
                                pu = pair.upper()
                                for k, v in raw.items():
                                    if pu in k.upper():
                                        ticker = v
                                        break
                            if ticker is None:
                                ticker = next(extra, None)
                            if ticker:
                                data[sym] = ticker
                            else:
                                logger.warning("Empty ticker result for %s", sym)
                except Exception:  # pragma: no cover - network
                    telemetry.inc("scan.api_errors")
                    data = {}
        else:
            try:
                pairs = [_id_for_symbol(exchange, s) for s in symbols]

                try:
                    async with TICKER_SEMA:
                        resp = await _fetch_ticker_async(pairs, timeout=timeout)
                        if delay:
                            await asyncio.sleep(delay)
                except TypeError:
                    async with TICKER_SEMA:
                        resp = await _fetch_ticker_async(pairs)
                        if delay:
                            await asyncio.sleep(delay)
                raw = resp.get("result", {})
                if resp.get("error"):
                    logger.warning(
                        "Ticker API errors for %s: %s",
                        ", ".join(pairs),
                        "; ".join(resp["error"]),
                    )
                    try:
                        async with TICKER_SEMA:
                            raw = (
                                await _fetch_ticker_async(symbols, timeout=timeout, exchange=exchange)
                            ).get("result", {})
                            if delay:
                                await asyncio.sleep(delay)
                    except TypeError:
                        async with TICKER_SEMA:
                            raw = (
                                await _fetch_ticker_async(symbols, exchange=exchange)
                            ).get("result", {})
                            if delay:
                                await asyncio.sleep(delay)
                data = {}
                if len(raw) == len(symbols):
                    for sym, (_, ticker) in zip(symbols, raw.items()):
                        if ticker:
                            data[sym] = ticker
                        else:
                            logger.warning("Empty ticker result for %s", sym)
                else:
                    extra = iter(raw.values())
                    for sym, pair in zip(symbols, pairs):
                        ticker = raw.get(pair) or raw.get(pair.upper())
                        if ticker is None:
                            pu = pair.upper()
                            for k, v in raw.items():
                                if pu in k.upper():
                                    ticker = v
                                    break
                        if ticker is None:
                            ticker = next(extra, None)
                        if ticker:
                            data[sym] = ticker
                        else:
                            logger.warning("Empty ticker result for %s", sym)
            except Exception:  # pragma: no cover - network
                telemetry.inc("scan.api_errors")
                data = {}

        if data:
            for sym, ticker in data.items():
                if sym not in symbols:
                    continue
                if not ticker:
                    logger.warning("Empty ticker result for %s", sym)
                    continue
                ticker_cache[sym] = ticker.get("info", ticker)
                ticker_ts[sym] = now
            record_results({s: t for s, t in data.items() if s in symbols})
            return {
                s: t.get("info", t)
                for s, t in data.items()
                if s in symbols and t
            }

    result: dict[str, dict] = {}
    if getattr(getattr(exchange, "has", {}), "get", lambda _k: False)("fetchTicker"):
        attempted_syms.update(symbols)
        for sym in symbols:
            try:
                fetcher = getattr(exchange, "fetch_ticker", None)
                if asyncio.iscoroutinefunction(fetcher):
                    async with TICKER_SEMA:
                        ticker = await fetcher(sym)
                        if delay:
                            await asyncio.sleep(delay)
                else:
                    async with TICKER_SEMA:
                        ticker = await asyncio.to_thread(fetcher, sym)
                        if delay:
                            await asyncio.sleep(delay)
                if ticker:
                    result[sym] = ticker.get("info", ticker)
                else:
                    logger.warning("Empty ticker result for %s", sym)
            except ccxt.BadSymbol as exc:  # pragma: no cover - network
                logger.warning("fetch_ticker BadSymbol for %s: %s", sym, exc)
                telemetry.inc("scan.api_errors")
            except Exception as exc:  # pragma: no cover - network
                logger.warning(
                    "fetch_ticker failed for %s: %s", sym, exc, exc_info=True
                )
                logger.warning("fetch_ticker failed for %s: %s", sym, exc, exc_info=log_exc)
                telemetry.inc("scan.api_errors")
    if result:
        for sym, ticker in result.items():
            if sym not in symbols:
                continue
            if not ticker:
                logger.warning("Empty ticker result for %s", sym)
                continue
            ticker_cache[sym] = ticker
            ticker_ts[sym] = now
    record_results(result)
    elapsed = time.perf_counter() - start_time
    logger.debug(
        "_refresh_tickers fetched %d tickers in %.2fs",
        len(symbols),
        elapsed,
    )
    return result




def _history_in_cache(df: pd.DataFrame | None, days: int, seconds: int) -> bool:
    """Return ``True`` when ``df`` has at least ``days`` days of candles."""
    if df is None or df.empty:
        return False
    candles_needed = int((days * 86400) / seconds)
    return len(df) >= candles_needed


async def _bounded_score(
    exchange,
    symbol: str,
    volume_usd: float,
    change_pct: float,
    spread_pct: float,
    liquidity_score: float,
    cfg: dict,
    df: pd.DataFrame | None = None,
) -> tuple[str, float]:
    """Return ``(symbol, score)`` for ``symbol`` using the global semaphore.

    ``df`` provides recent OHLCV data for ``symbol``.  When supplied it is
    forwarded to :func:`score_symbol` so the raw score can be normalized by the
    symbol's volatility.  Passing ``None`` disables this normalization.
    """

    async with SEMA:
        score = await score_symbol(
            exchange,
            symbol,
            volume_usd,
            change_pct,
            spread_pct,
            liquidity_score,
            cfg,
            df,
        )
    return symbol, score


async def filter_symbols(
    exchange,
    symbols: Iterable[str],
    config: dict | None = None,
    df_cache: Dict[str, pd.DataFrame] | None = None,
) -> tuple[List[tuple[str, float]], List[tuple[str, float]]]:
    """Return CEX symbols and onchain symbols with basic scoring."""

    start_time = time.perf_counter()

    cfg = config or {}
    sf = cfg.get("symbol_filter", {})
    init_semaphore(sf.get("max_concurrent_ohlcv", cfg.get("max_concurrent_ohlcv", 4)))
    init_ticker_semaphore(
        sf.get("max_concurrent_tickers", cfg.get("max_concurrent_tickers")),
        cfg.get("ticker_rate_limit"),
    )
    min_volume = sf.get("min_volume_usd", DEFAULT_MIN_VOLUME_USD)
    vol_pct = sf.get("volume_percentile", DEFAULT_VOLUME_PERCENTILE)
    max_spread = sf.get("max_spread_pct", 1.0)
    pct = sf.get("change_pct_percentile", DEFAULT_CHANGE_PCT_PERCENTILE)
    cache_map = load_liquid_map()
    vol_mult_default = 1 if cache_map is None else 2
    vol_mult = sf.get("uncached_volume_multiplier", vol_mult_default)
    min_age = cfg.get("min_symbol_age_days", 0)
    min_score = float(cfg.get("min_symbol_score", 0.0))
    cache_changed = False

    symbols = list(symbols)
    cex_syms = [s for s in symbols if not str(s).upper().endswith("/USDC")]
    onchain_syms = [s for s in symbols if str(s).upper().endswith("/USDC")]

    telemetry.inc("scan.symbols_considered", len(symbols))
    skipped = 0

    cached_data: dict[str, tuple[float, float, float]] = {}
    for sym in cex_syms:
        if sym in liq_cache:
            cached_data[sym] = liq_cache[sym]

    try:
        data = await _refresh_tickers(exchange, cex_syms, cfg)
    except Exception:
        raise

    # map of ids returned by Kraken to human readable symbols
    id_map: dict[str, str] = {}
    request_map = {_norm_symbol(s.replace("/", "")): _norm_symbol(s) for s in cex_syms}

    if hasattr(exchange, "markets_by_id"):
        if not exchange.markets_by_id and hasattr(exchange, "load_markets"):
            try:
                exchange.load_markets()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("load_markets failed: %s", exc)
        for k, v in exchange.markets_by_id.items():
            if isinstance(v, dict):
                symbol = _norm_symbol(v.get("symbol", k))
                id_map[k] = symbol
                alt = (
                    v.get("altname")
                    or v.get("wsname")
                    or v.get("info", {}).get("altname")
                    or v.get("info", {}).get("wsname")
                )
                if alt:
                    alt_id = alt.replace("/", "").upper()
                    id_map[alt] = symbol
                    id_map[alt_id] = symbol
                    if len(alt_id) in (6, 7):
                        id_map.setdefault(
                            _norm_symbol(f"{alt_id[:-3]}/{alt_id[-3:]}"), symbol
                        )
                    elif len(alt_id) == 8 and alt_id[0] in "XZ" and alt_id[4] in "XZ":
                        id_map.setdefault(
                            _norm_symbol(f"{alt_id[1:4]}/{alt_id[5:]}"), symbol
                        )
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                id_map[k] = _norm_symbol(v[0].get("symbol", k))
            else:
                id_map[k] = _norm_symbol(v if isinstance(v, str) else k)

    for req_id, sym in request_map.items():
        id_map.setdefault(req_id, sym)

    metrics: List[tuple[str, float, float, float]] = []

    raw: List[tuple[str, float, float, float]] = []
    volumes: List[float] = []
    seen: set[str] = set()
    for pair_id, ticker in data.items():
        symbol = id_map.get(pair_id) or id_map.get(pair_id.upper())
        norm = pair_id.upper()
        if not symbol:
            if norm.startswith("X"):
                norm = norm[1:]
            norm = norm.replace("ZUSD", "USD").replace("ZUSDT", "USDT")
            norm = _norm_symbol(norm)
            symbol = id_map.get(norm)
        if not symbol and "/" in pair_id:
            symbol = _norm_symbol(pair_id)
        if not symbol:
            symbol = request_map.get(norm)
        if not symbol:
            for req_id, sym in request_map.items():
                if norm.endswith(req_id):
                    symbol = sym
                    break
        if not symbol:
            logger.warning("Unable to map ticker id %s to requested symbol", pair_id)
            continue

        vol_usd, change_pct, spread_pct = await _parse_metrics(exchange, symbol, ticker)
        logger.debug(
            "Ticker %s volume %.2f USD change %.2f%% spread %.2f%%",
            symbol,
            vol_usd,
            change_pct,
            spread_pct,
        )
        seen.add(symbol)
        local_min_volume = min_volume * 0.5 if symbol.endswith("/USDC") else min_volume
        if vol_usd < local_min_volume:
            logger.warning(
                "Skipping %s due to low ticker volume %.2f < %.2f",
                symbol,
                vol_usd,
                local_min_volume,
            )
            skipped += 1
            continue
        if cache_map and vol_usd < local_min_volume * vol_mult:
            skipped += 1
            continue
        if vol_usd >= local_min_volume and spread_pct <= max_spread:
            metrics.append((symbol, vol_usd, change_pct, spread_pct))
            if cache_map is not None and symbol not in cache_map:
                cache_map[symbol] = time.time()
                cache_changed = True
        if vol_usd >= local_min_volume:
            volumes.append(vol_usd)
            raw.append((symbol, vol_usd, change_pct, spread_pct))

    for sym in cex_syms:
        norm_sym = _norm_symbol(sym)
        if norm_sym in seen:
            continue
        cached = cached_data.get(sym) or cached_data.get(norm_sym)
        if cached is None:
            logger.warning(
                "No ticker data returned for %s; skipping",
                sym,
            )
            skipped += 1
            continue
        vol_usd, spread_pct, _ = cached
        seen.add(norm_sym)
        local_min_volume = min_volume * 0.5 if norm_sym.endswith("/USDC") else min_volume
        if cache_map and vol_usd < local_min_volume * vol_mult:
            skipped += 1
            continue
        if vol_usd >= local_min_volume and spread_pct <= max_spread:
            metrics.append((norm_sym, vol_usd, 0.0, spread_pct))
            if cache_map is not None and norm_sym not in cache_map:
                cache_map[norm_sym] = time.time()
                cache_changed = True
        if vol_usd >= local_min_volume:
            volumes.append(vol_usd)
            raw.append((norm_sym, vol_usd, 0.0, spread_pct))

    vol_cut = np.percentile(volumes, vol_pct) if volumes else 0

    metrics: List[tuple[str, float, float, float]] = []
    for sym, vol_usd, change_pct, spread_pct in raw:
        if vol_usd >= vol_cut and spread_pct <= max_spread:
            metrics.append((sym, vol_usd, change_pct, spread_pct))
        else:
            skipped += 1

    candidates = [m[0] for m in metrics if m[1] >= vol_cut]
    if candidates:
        if df_cache is None:
            df_cache = {}
        ohlcv_bs = cfg.get("ohlcv_batch_size")
        if ohlcv_bs is None:
            ohlcv_bs = sf.get("ohlcv_batch_size")
        df_cache = await update_multi_tf_ohlcv_cache(
            exchange,
            {"1h": df_cache} if isinstance(df_cache, dict) and "1h" not in df_cache else df_cache,
            candidates,
            {
                "timeframes": ["1h", "4h", "1d"],
                "ohlcv_batch_size": ohlcv_bs,
            },
            limit=int(sf.get("initial_history_candles", 300)),
            max_concurrent=min(10, len(candidates)),
            batch_size=ohlcv_bs,
        )
        if "1h" in df_cache:
            df_cache = df_cache.get("1h", {})

    if metrics and pct:
        threshold = np.percentile([abs(m[2]) for m in metrics], pct)
        metrics = [m for m in metrics if abs(m[2]) >= threshold]

    liq_scores: Dict[str, float] = {}
    if metrics:
        denom = float(cfg.get("trade_size_pct", 0.1)) * float(cfg.get("balance", 1))
        if denom <= 0:
            denom = 1.0

        async def _fetch_liq(sym: str) -> tuple[str, float]:
            base, _, quote = sym.partition("/")
            is_solana = quote.upper() == "USDC" and base.upper() not in NON_SOLANA_BASES
            if not is_solana:
                return sym, 0.0
            try:
                _, _, reserve = await fetch_geckoterminal_ohlcv(sym, limit=1)
                return sym, reserve / denom
            except Exception:
                return sym, 0.0

        liq_results = await asyncio.gather(
            *[_fetch_liq(sym) for sym, *_ in metrics if sym.upper().endswith("/USDC")]
        )
        liq_scores = {s: sc for s, sc in liq_results}

    scored: List[tuple[str, float]] = []
    if metrics:
        results = await asyncio.gather(
            *[
                _bounded_score(
                    exchange,
                    sym,
                    vol,
                    chg,
                    spr,
                    liq_scores.get(sym, 1.0),
                    cfg,
                    df_cache.get(sym) if df_cache else None,
                )
                for sym, vol, chg, spr in metrics
            ]
        )
        for sym, score in results:
            if score >= min_score:
                scored.append((sym, score))
            else:
                skipped += 1
    scored.sort(key=lambda x: x[1], reverse=True)

    corr_map: Dict[tuple[str, str], float] = {}
    if df_cache:
        # only compute correlations for the top N scoring symbols
        max_pairs = sf.get("correlation_max_pairs")
        if max_pairs:
            top = [s for s, _ in scored[:max_pairs]]
        else:
            top = [s for s, _ in scored]
        subset = {s: df_cache.get(s) for s in top}
        window = sf.get("correlation_window", 30)
        have_history = all(
            isinstance(df, pd.DataFrame) and len(df) >= window for df in subset.values()
        )
        if have_history:
            corr_map = incremental_correlation(subset, window=window)
        else:
            corr_map = {}

    seconds = timeframe_seconds(exchange, "1h") if min_age > 0 else 0
    if min_age > 0:
        missing = [
            s
            for s, _ in scored
            if not _history_in_cache(
                df_cache.get(s) if df_cache else None, min_age, seconds
            )
        ]
        if missing:
            if df_cache is None:
                df_cache = {}
            history_ok = await asyncio.gather(
                *[has_enough_history(exchange, s, min_age, "1h") for s in missing]
            )
            candles_needed = int((min_age * 86400) / seconds)
            for sym, ok in zip(missing, history_ok):
                if ok:
                    df_cache[sym] = pd.DataFrame({"close": [0] * candles_needed})
            missing = [s for s, ok in zip(missing, history_ok) if not ok]
            if missing:
                df_cache = await update_ohlcv_cache(
                    exchange,
                    df_cache,
                    missing,
                    timeframe="1h",
                    limit=int((min_age * 86400) / seconds),
                    max_concurrent=len(missing),
                    config=cfg,
                )

    result: List[tuple[str, float]] = []
    for sym, score in scored:
        if min_age > 0 and not _history_in_cache(
            df_cache.get(sym) if df_cache else None, min_age, seconds
        ):
            logger.debug("Skipping %s due to insufficient history", sym)
            skipped += 1
            continue
        keep = True
        if df_cache:
            for kept, _ in result:
                corr = corr_map.get((sym, kept)) or corr_map.get((kept, sym))
                if corr is not None and corr >= 0.95:
                    keep = False
                    break
        if keep:
            logger.info("Selected %s with score %.2f", sym, score)
            result.append((sym, score))
        else:
            skipped += 1

    telemetry.inc("scan.symbols_skipped", skipped)
    if cache_changed and cache_map is not None:
        try:
            Path(PAIR_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(PAIR_FILE, "w") as f:
                json.dump(cache_map, f, indent=2)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to update %s: %s", PAIR_FILE, exc)

    from crypto_bot.utils.token_registry import (
        TOKEN_MINTS,
        get_mint_from_gecko,
        fetch_from_helius,
    )

    resolved_onchain: List[tuple[str, float]] = []
    onchain_min_volume = cfg.get("onchain_min_volume_usd", 10_000_000)
    for sym in onchain_syms:
        base, _, quote = sym.partition("/")
        is_solana = quote.upper() == "USDC" and base.upper() not in NON_SOLANA_BASES
        if not is_solana:
            continue
        base = base.upper()
        mint = TOKEN_MINTS.get(base)
        if not mint:
            logger.debug("No mint for %s; attempting lookup", sym)
            mint = await get_mint_from_gecko(base)
            if not mint:
                helius = await fetch_from_helius([base])
                mint = helius.get(base.upper()) if helius else None
            if mint:
                TOKEN_MINTS[base] = mint
            else:
                logger.warning(
                    "Mint lookup failed for %s - consider adding to TOKEN_MINTS or NON_SOLANA_BASES",
                    sym,
                )
                continue
        logger.info("Resolved %s to mint %s for onchain", sym, mint)
        try:
            _, vol, _ = await fetch_geckoterminal_ohlcv(sym, limit=1)
            if vol < onchain_min_volume:
                logger.info(
                    "Skipping %s due to low volume %.2f USD (min %.2f)",
                    sym,
                    vol,
                    onchain_min_volume,
                )
                continue
            score = vol / float(onchain_min_volume)
            resolved_onchain.append((sym, score))
        except Exception:  # pragma: no cover - network
            logger.warning("Gecko fetch failed for %s; skipping", sym)

    elapsed = time.perf_counter() - start_time
    logger.debug(
        "filter_symbols processed %d symbols in %.2fs",
        len(symbols),
        elapsed,
    )
    return (
        sorted(result, key=lambda x: x[1], reverse=True),
        sorted(resolved_onchain, key=lambda x: x[1], reverse=True),
    )

"""Utility for filtering trading pairs based on liquidity and volatility."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Dict

import ccxt

import aiohttp
import numpy as np
import pandas as pd
import yaml
from cachetools import TTLCache

from .logger import LOG_DIR, setup_logger
from .market_loader import (
    fetch_ohlcv_async,
    update_ohlcv_cache,
    fetch_geckoterminal_ohlcv,
)
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
SEMA = asyncio.Semaphore(cfg.get("max_concurrent_ohlcv", 4))


logger = setup_logger(__name__, LOG_DIR / "symbol_filter.log")

API_URL = "https://api.kraken.com/0/public"
DEFAULT_MIN_VOLUME_USD = 50000
DEFAULT_VOLUME_PERCENTILE = 30
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

# Cache of recent liquidity metrics per symbol
liq_cache = TTLCache(maxsize=2000, ttl=900)


async def has_enough_history(
    exchange, symbol: str, days: int = 30, timeframe: str = "1d"
) -> bool:
    """Return ``True`` when ``symbol`` has at least ``days`` days of history."""
    seconds = _timeframe_seconds(exchange, timeframe)
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
        logger.warning(
            "fetch_ohlcv returned exception for %s on %s for %d days: %s",
            symbol,
            timeframe,
            days,
            exc,
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
    if not isinstance(data, list) or len(data) < candles_needed:
        return False
    return True


async def _fetch_ticker_async(pairs: Iterable[str], timeout: int = 10) -> dict:
    """Return ticker data for ``pairs`` in batches of 20 using aiohttp."""

    mock = os.getenv("MOCK_KRAKEN_TICKER")
    if mock:
        return json.loads(mock)

    pairs_list = list(pairs)
    combined: dict = {"result": {}, "error": []}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(pairs_list), 20):
            chunk = pairs_list[i : i + 20]
            url = f"{API_URL}/Ticker?pair={','.join(chunk)}"
            tasks.append(session.get(url, timeout=timeout))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            resp.raise_for_status()
            data = await resp.json()
            combined["error"] += data.get("error", [])
            combined["result"].update(data.get("result", {}))

    return combined


def _parse_metrics(symbol: str, ticker: dict) -> tuple[float, float, float]:
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

    cfg = config if config is not None else globals().get("cfg", {})
    sf = cfg.get("symbol_filter", {})
    attempts = int(sf.get("ticker_retry_attempts", 3))
    if attempts < 1:
        attempts = 1
    log_exc = sf.get("log_ticker_exceptions", False)

    now = time.time()
    batch = cfg.get("symbol_filter", {}).get("kraken_batch_size", 100)
    timeout = cfg.get("symbol_filter", {}).get("http_timeout", 10)
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
                "Symbols not in exchange.markets: %s",
                ", ".join(missing),
            )

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
            if ws_failures:
                await asyncio.sleep(min(2 ** (ws_failures - 1), 30))
            try:
                data = await exchange.watch_tickers(to_fetch)
                exchange.options["ws_failures"] = 0
            except Exception as exc:  # pragma: no cover - network
                logger.warning("watch_tickers failed: %s", exc, exc_info=log_exc)
                logger.info("watch_tickers failed, falling back to HTTP fetch")
                telemetry.inc("scan.api_errors")
                failures = exchange.options.get("ws_failures", 0) + 1
                exchange.options["ws_failures"] = failures
                if failures >= ws_limit:
                    exchange.options["ws_scan"] = False
                telemetry.inc("scan.ws_errors")
                try_ws = False
                try_http = True
        for sym, ticker in data.items():
            ticker_cache[sym] = ticker.get("info", ticker)
            ticker_ts[sym] = now
        result = {s: ticker_cache[s] for s in symbols if s in ticker_cache}
        if result:
            return {s: t.get("info", t) for s, t in result.items()}

    if try_http:
        if getattr(getattr(exchange, "has", {}), "get", lambda _k: False)(
            "fetchTickers"
        ):
            data = {}
            symbols_list = list(symbols)
            for i in range(0, len(symbols_list), batch):
                chunk = symbols_list[i : i + batch]
                chunk_data: dict | None = None
                for attempt in range(3):
                    try:
                        fetcher = getattr(exchange, "fetch_tickers", None)
                        if asyncio.iscoroutinefunction(fetcher):
                            fetched = await fetcher(chunk)
                        else:
                            fetched = await asyncio.to_thread(fetcher, chunk)
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
                    except (ccxt.ExchangeError, ccxt.NetworkError) as exc:  # pragma: no cover - network
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
                                fetched = await fetcher(list(symbols))
                            else:
                                fetched = await asyncio.to_thread(fetcher, list(symbols))
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
                            ccxt.NetworkError,
                        ) as exc:  # pragma: no cover - network
                            if getattr(exc, "http_status", None) in (520, 522) and attempt < 2:
                                await asyncio.sleep(2**attempt)
                        except (ccxt.ExchangeError, ccxt.NetworkError) as exc:  # pragma: no cover - network
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
                    pairs = [s.replace("/", "") for s in symbols]
                    try:
                        raw = (await _fetch_ticker_async(pairs, timeout=timeout)).get("result", {})
                    except TypeError:
                        raw = (await _fetch_ticker_async(pairs)).get("result", {})
                    data = {}
                    if len(raw) == len(symbols):
                        for sym, (_, ticker) in zip(symbols, raw.items()):
                            data[sym] = ticker
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
                            if ticker is not None:
                                data[sym] = ticker
                except Exception:  # pragma: no cover - network
                    telemetry.inc("scan.api_errors")
                    data = {}
        else:
            try:
                pairs = [s.replace("/", "") for s in symbols]
                try:
                    raw = (await _fetch_ticker_async(pairs, timeout=timeout)).get("result", {})
                except TypeError:
                    raw = (await _fetch_ticker_async(pairs)).get("result", {})
                data = {}
                if len(raw) == len(symbols):
                    for sym, (_, ticker) in zip(symbols, raw.items()):
                        data[sym] = ticker
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
                        if ticker is not None:
                            data[sym] = ticker
            except Exception:  # pragma: no cover - network
                telemetry.inc("scan.api_errors")
                data = {}

        if data:
            for sym, ticker in data.items():
                ticker_cache[sym] = ticker.get("info", ticker)
                ticker_ts[sym] = now
            return {s: t.get("info", t) for s, t in data.items()}

    result: dict[str, dict] = {}
    if getattr(getattr(exchange, "has", {}), "get", lambda _k: False)("fetchTicker"):
        for sym in symbols:
            try:
                fetcher = getattr(exchange, "fetch_ticker", None)
                if asyncio.iscoroutinefunction(fetcher):
                    ticker = await fetcher(sym)
                else:
                    ticker = await asyncio.to_thread(fetcher, sym)
                result[sym] = ticker.get("info", ticker)
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
            ticker_cache[sym] = ticker
            ticker_ts[sym] = now
    return result


def _timeframe_seconds(exchange, timeframe: str) -> int:
    if hasattr(exchange, "parse_timeframe"):
        try:
            return exchange.parse_timeframe(timeframe)
        except Exception:
            pass
    unit = timeframe[-1]
    value = int(timeframe[:-1])
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
) -> tuple[str, float]:
    """Return ``(symbol, score)`` using the global semaphore."""

    async with SEMA:
        score = await score_symbol(
            exchange, symbol, volume_usd, change_pct, spread_pct, liquidity_score, cfg
        )
    return symbol, score


async def filter_symbols(
    exchange,
    symbols: Iterable[str],
    config: dict | None = None,
    df_cache: Dict[str, pd.DataFrame] | None = None,
) -> List[tuple[str, float]]:
    """Return ``symbols`` passing liquidity checks sorted by score."""

    cfg = config or {}
    sf = cfg.get("symbol_filter", {})
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

    telemetry.inc("scan.symbols_considered", len(list(symbols)))
    skipped = 0

    cached_data: dict[str, tuple[float, float, float]] = {}
    for sym in symbols:
        if sym in liq_cache:
            cached_data[sym] = liq_cache[sym]

    try:
        data = await _refresh_tickers(exchange, symbols, cfg)
    except Exception:
        raise

    # map of ids returned by Kraken to human readable symbols
    id_map: dict[str, str] = {}
    request_map = {_norm_symbol(s.replace("/", "")): _norm_symbol(s) for s in symbols}

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

        vol_usd, change_pct, spread_pct = _parse_metrics(symbol, ticker)
        logger.debug(
            "Ticker %s volume %.2f USD change %.2f%% spread %.2f%%",
            symbol,
            vol_usd,
            change_pct,
            spread_pct,
        )
        seen.add(symbol)
        local_min_volume = min_volume * 0.5 if symbol.endswith("/USDC") else min_volume
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

    for sym in symbols:
        norm_sym = _norm_symbol(sym)
        if norm_sym in seen:
            continue
        cached = cached_data.get(sym) or cached_data.get(norm_sym)
        if cached is None:
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

    if metrics and pct:
        threshold = np.percentile([abs(m[2]) for m in metrics], pct)
        metrics = [m for m in metrics if abs(m[2]) >= threshold]

    liq_scores: Dict[str, float] = {}
    if metrics:
        denom = float(cfg.get("trade_size_pct", 0.1)) * float(cfg.get("balance", 1))
        if denom <= 0:
            denom = 1.0

        async def _fetch_liq(sym: str) -> tuple[str, float]:
            try:
                _, _, reserve = await fetch_geckoterminal_ohlcv(sym, limit=1)
                return sym, reserve / denom
            except Exception:
                return sym, 0.0

        liq_results = await asyncio.gather(
            *[_fetch_liq(sym) for sym, *_ in metrics if sym.endswith("/USDC")]
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

    seconds = _timeframe_seconds(exchange, "1h") if min_age > 0 else 0
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
            df_cache = await update_ohlcv_cache(
                exchange,
                df_cache,
                missing,
                timeframe="1h",
                limit=int((min_age * 86400) / seconds),
                max_concurrent=len(missing),
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
    return result

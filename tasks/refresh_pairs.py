from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
import logging
import os

import ccxt.async_support as ccxt
import yaml
import aiohttp
from crypto_bot.utils.market_loader import timeframe_seconds
from crypto_bot.utils.symbol_utils import fix_symbol

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
PAIR_FILE = CACHE_DIR / "liquid_pairs.json"

DEFAULT_MIN_VOLUME_USD = 10_000_000
DEFAULT_TOP_K = 40
DEFAULT_REFRESH_INTERVAL = 6 * 3600  # 6 hours

logger = logging.getLogger(__name__)


def is_cache_fresh() -> bool:
    """Return ``True`` if ``PAIR_FILE`` exists and is newer than one hour."""
    if not PAIR_FILE.exists():
        return False
    try:
        mtime = PAIR_FILE.stat().st_mtime
    except OSError:
        return False
    return time.time() - mtime < 3600


def _parse_interval(value: str | int | float) -> float:
    """Return ``value`` in seconds, accepting shorthand like "6h"."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return float(timeframe_seconds(None, value))
            except Exception:
                pass
    return float(DEFAULT_REFRESH_INTERVAL)


def load_config() -> dict:
    """Load YAML configuration if available."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        with open(trend_file) as sf:
            overrides = yaml.safe_load(sf) or {}
        trend_cfg = data.get("trend_bot", {})
        if isinstance(trend_cfg, dict):
            trend_cfg.update(overrides)
        else:
            trend_cfg = overrides
        data["trend_bot"] = trend_cfg

    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]

    trading_cfg = data.get("trading", {}) or {}
    raw_ex = data.get("exchange") or trading_cfg.get("exchange") or os.getenv("EXCHANGE")
    if isinstance(raw_ex, dict):
        ex_cfg = dict(raw_ex)
    else:
        ex_cfg = {"name": raw_ex}
    ex_cfg.setdefault("name", "kraken")
    ex_cfg.setdefault("max_concurrency", 3)
    ex_cfg.setdefault("request_timeout_ms", 10000)
    data["exchange"] = ex_cfg

    return data


def get_exchange(config: dict) -> ccxt.Exchange:
    """Instantiate the configured ccxt exchange."""
    ex = config.get("exchange", "kraken")
    if isinstance(ex, dict):
        name = ex.get("name", "kraken").lower()
        params = {"enableRateLimit": True}
        timeout = ex.get("request_timeout_ms")
        if timeout:
            params["timeout"] = int(timeout)
        max_conc = ex.get("max_concurrency")
    else:
        name = str(ex).lower()
        params = {"enableRateLimit": True}
        max_conc = None
    if not hasattr(ccxt, name):
        raise ValueError(f"Unsupported exchange: {name}")
    exchange = getattr(ccxt, name)(params)
    if max_conc is not None:
        setattr(exchange, "max_concurrency", int(max_conc))
    return exchange


async def _fetch_tickers(exchange: ccxt.Exchange) -> dict:
    """Fetch tickers with a 10 second timeout."""
    return await asyncio.wait_for(exchange.fetch_tickers(), 10)


async def _close_exchange(exchange: ccxt.Exchange) -> None:
    close = getattr(exchange, "close", None)
    if close:
        try:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                close()
        except Exception:  # pragma: no cover - best effort
            pass


async def get_solana_liquid_pairs(min_volume: float, quote: str = "USDC") -> list[str]:
    """Return Raydium symbols with liquidity above ``min_volume`` using ``quote`` as quote currency."""
    url = "https://api.raydium.io/v2/main/pairs"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Failed to fetch Solana pairs: %s", exc)
        return []

    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    quote = quote.upper()
    results: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            base_sym = item.get("base", {}).get("symbol")
            quote_sym = item.get("quote", {}).get("symbol")
            if isinstance(base_sym, str) and isinstance(quote_sym, str):
                name = f"{base_sym}/{quote_sym}"
            else:
                continue
        base, _, q = name.partition("/")
        if q.upper() != quote or not base:
            continue
        vol = (
            item.get("liquidity")
            or item.get("liquidityUsd")
            or item.get("liquidity_usd")
            or item.get("liquidityUSD")
            or item.get("volumeUsd")
            or item.get("volume_usd")
            or item.get("volume24hQuote")
            or item.get("volume24h")
            or 0.0
        )
        try:
            amount = float(vol)
        except Exception:
            amount = 0.0
        if amount >= min_volume:
            results.append(f"{base.upper()}/{quote}")

    return results


async def refresh_pairs_async(
    min_volume_usd: float, top_k: int, config: dict, *, force_refresh: bool = False
) -> list[str]:
    """Fetch tickers and update the cached liquid pairs list."""
    if not force_refresh and is_cache_fresh():
        try:
            with open(PAIR_FILE) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
            if isinstance(loaded, dict):
                return list(loaded)
        except Exception as exc:  # pragma: no cover - corrupted cache
            logger.error("Failed to read %s: %s", PAIR_FILE, exc)

    old_pairs: list[str] = []
    old_map: dict[str, float] = {}
    mtime = 0.0
    if PAIR_FILE.exists():
        try:
            with open(PAIR_FILE) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    old_pairs = loaded
                elif isinstance(loaded, dict):
                    old_pairs = list(loaded)
                    old_map = {k: float(v) for k, v in loaded.items()}
            mtime = PAIR_FILE.stat().st_mtime
        except Exception as exc:  # pragma: no cover - corrupted cache
            logger.error("Failed to read %s: %s", PAIR_FILE, exc)

    rp_cfg = config.get("refresh_pairs", {}) if isinstance(config, dict) else {}
    refresh_interval = _parse_interval(
        rp_cfg.get("refresh_interval", DEFAULT_REFRESH_INTERVAL)
    )
    onchain_quote = str(config.get("onchain_default_quote", "USDC")).upper()
    allowed_quotes = {
        q.upper() for q in rp_cfg.get("allowed_quote_currencies", []) if isinstance(q, str)
    }
    blacklist = {
        a.upper() for a in rp_cfg.get("blacklist_assets", []) if isinstance(a, str)
    }

    if old_pairs and mtime and time.time() - mtime < refresh_interval and not force_refresh:
        return old_pairs

    exchange = get_exchange(config)
    sec_name = config.get("refresh_pairs", {}).get("secondary_exchange")
    secondary = get_exchange({"exchange": sec_name}) if sec_name else None
    try:
        tasks = [_fetch_tickers(exchange)]
        if secondary:
            tasks.append(_fetch_tickers(secondary))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        primary_res = results[0]
        if isinstance(primary_res, Exception):
            raise primary_res
        tickers = primary_res
        if not isinstance(tickers, dict):
            raise TypeError("fetch_tickers returned invalid data")
        if secondary:
            sec_res = results[1]
            if not isinstance(sec_res, Exception) and isinstance(sec_res, dict):
                for sym, data in sec_res.items():
                    vol2 = data.get("quoteVolume")
                    if sym in tickers:
                        vol1 = tickers[sym].get("quoteVolume")
                        if vol2 is not None and (vol1 is None or float(vol2) > float(vol1)):
                            tickers[sym] = data
                    else:
                        tickers[sym] = data
        sol_pairs = await get_solana_liquid_pairs(min_volume_usd, onchain_quote)
        for sym in sol_pairs:
            tickers.setdefault(sym, {"quoteVolume": min_volume_usd})
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch tickers: %s", exc)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PAIR_FILE, "w") as f:
            json.dump(old_map or {p: time.time() for p in old_pairs}, f, indent=2)
        return old_pairs
    finally:
        await _close_exchange(exchange)
        if secondary:
            await _close_exchange(secondary)

    pairs: list[tuple[str, float]] = []
    for symbol, data in tickers.items():
        vol = data.get("quoteVolume")
        if vol is None:
            continue

        parts = symbol.split("/")
        if len(parts) != 2:
            continue
        base, quote = parts[0].upper(), parts[1].upper()

        if allowed_quotes and quote not in allowed_quotes:
            continue
        if base in blacklist:
            continue

        pairs.append((symbol, float(vol)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top_list = [sym for sym, vol in pairs if vol >= min_volume_usd][:top_k]
    top_map = {sym: time.time() for sym in top_list}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIR_FILE, "w") as f:
        json.dump(top_map, f, indent=2)

    return top_list


def refresh_pairs(
    min_volume_usd: float, top_k: int, config: dict, *, force_refresh: bool = False
) -> list[str]:
    """Synchronous wrapper for :func:`refresh_pairs_async`."""
    return asyncio.run(
        refresh_pairs_async(min_volume_usd, top_k, config, force_refresh=force_refresh)
    )


def main() -> None:
    cfg = load_config()
    rp_cfg = cfg.get("refresh_pairs", {})

    parser = argparse.ArgumentParser(description="Refresh liquid trading pairs")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--min-quote-volume-usd",
        type=float,
        default=float(rp_cfg.get("min_quote_volume_usd", DEFAULT_MIN_VOLUME_USD)),
    )
    parser.add_argument("--top-k", type=int, default=int(rp_cfg.get("top_k", DEFAULT_TOP_K)))
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=_parse_interval(rp_cfg.get("refresh_interval", DEFAULT_REFRESH_INTERVAL)),
        help="Refresh interval in seconds",
    )
    args = parser.parse_args()

    while True:
        try:
            pairs = refresh_pairs(args.min_quote_volume_usd, args.top_k, cfg)
            print(f"Updated {PAIR_FILE} with {len(pairs)} pairs")
        except Exception as exc:  # pragma: no cover - network failures
            print(f"Failed to refresh pairs: {exc}")
        if args.once:
            break
        time.sleep(args.refresh_interval)


if __name__ == "__main__":
    main()

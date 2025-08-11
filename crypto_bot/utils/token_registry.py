from __future__ import annotations

import json
import logging
import os
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, Iterable, List

import aiohttp
import ccxt.async_support as ccxt

from crypto_bot.strategy import cross_chain_arb_bot
from .gecko import gecko_request

try:  # optional dependency
    from coinTrader_Trainer.ml_trainer import fetch_data_range_async  # type: ignore
except Exception:  # pragma: no cover - optional
    async def fetch_data_range_async(*args, **kwargs):  # type: ignore
        return None

logger = logging.getLogger(__name__)

TOKEN_REGISTRY_URL = "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"

# Primary token list from Jupiter API
# ``station.jup.ag`` now redirects to ``dev.jup.ag`` which returns ``404``.
# The latest stable token list is hosted at ``https://token.jup.ag/all``.
JUPITER_TOKEN_URL = "https://token.jup.ag/all"

# Batch metadata endpoint for resolving unknown symbols
HELIUS_TOKEN_API = "https://api.helius.xyz/v0/token-metadata"

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "token_mints.json"

# Mapping of token symbols to Solana mints. ``load_token_mints`` populates this
# dictionary at runtime.
TOKEN_MINTS: Dict[str, str] = {}

# Mapping of mint addresses to their decimal precision
TOKEN_DECIMALS: Dict[str, int] = {}

_LOADED = False

PUMP_URL = "https://api.pump.fun/tokens?limit=50&offset=0"
RAYDIUM_URL = "https://api.raydium.io/v2/main/pairs"

# Poll interval for monitoring external token feeds
# Reduced poll interval to surface new tokens faster
POLL_INTERVAL = 10


def to_base_units(amount_tokens: float, decimals: int) -> int:
    """Convert human readable ``amount_tokens`` to integer base units."""
    factor = Decimal(10) ** decimals
    quantized = (Decimal(str(amount_tokens)) * factor).quantize(
        Decimal("1"), rounding=ROUND_DOWN
    )
    return int(quantized)


async def fetch_from_jupiter() -> Dict[str, str]:
    """Return mapping of symbols to mints using Jupiter token list."""
    async with aiohttp.ClientSession() as session:
        async with session.get(JUPITER_TOKEN_URL, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    result: Dict[str, str] = {}
    tokens = data if isinstance(data, list) else data.get("tokens") or []
    for item in tokens:
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("address") or item.get("mint") or item.get("tokenMint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
            dec = item.get("decimals")
            if isinstance(dec, int):
                TOKEN_DECIMALS[mint] = dec
    return result


async def _check_cex_arbitrage(symbol: str) -> None:
    """Check Kraken and Coinbase for arbitrage on ``symbol`` and trade to BTC."""

    pair = f"{symbol}/USD"
    try:
        kraken = ccxt.kraken()
        coinbase = ccxt.coinbase()
        ticker_kraken, ticker_coinbase = await asyncio.gather(
            kraken.fetch_ticker(pair), coinbase.fetch_ticker(pair)
        )
    except Exception as exc:  # pragma: no cover - network
        logger.error("CEX fetch failed for %s: %s", pair, exc)
        return
    finally:
        try:
            await kraken.close()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            await coinbase.close()
        except Exception:  # pragma: no cover - best effort
            pass

    p1 = ticker_kraken.get("last") or ticker_kraken.get("close")
    p2 = ticker_coinbase.get("last") or ticker_coinbase.get("close")
    if p1 is None or p2 is None:
        return
    try:
        f1 = float(p1)
        f2 = float(p2)
    except Exception:
        return
    if f1 <= 0 or f2 <= 0:
        return
    spread = abs(f1 - f2) / ((f1 + f2) / 2)
    if spread <= 0.005:
        return

    exec_fn = getattr(cross_chain_arb_bot, "execute_arbitrage", None)
    if exec_fn is None:
        return
    try:
        await exec_fn(pair, "BTC")
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Arbitrage execution failed for %s: %s", pair, exc)


async def _run_ml_trainer() -> None:
    """Invoke the ML trainer asynchronously."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "ml_trainer.py", "train", "regime", "--use-gpu"
        )
        await proc.wait()
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("ml_trainer failed: %s", exc)


async def _fetch_and_train(start: datetime, end: datetime) -> None:
    """Fetch training data and kick off training in the background."""
    try:
        await fetch_data_range_async("trade_logs", start.isoformat(), end.isoformat())
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("fetch_data_range_async failed: %s", exc)
    await _run_ml_trainer()


async def load_token_mints(
    url: str | None = None,
    *,
    unknown: List[str] | None = None,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """Return mapping of token symbols to mint addresses.

    The list is fetched from ``url`` or ``TOKEN_MINTS_URL`` environment variable.
    Results are cached on disk and subsequent calls return an empty dict unless
    ``force_refresh`` is ``True``.
    The Solana list is fetched from Jupiter first then GitHub as a fallback.
    Cached results are reused unless ``force_refresh`` is ``True``.
    Unknown ``symbols`` can be resolved via the Helius API.
    """
    global _LOADED
    if _LOADED and not force_refresh:
        return {}

    mapping: Dict[str, str] = {}

    # Fetch from Jupiter with retries
    for attempt in range(3):
        try:
            mapping = await fetch_from_jupiter()
            if mapping:
                break
        except Exception as exc:  # pragma: no cover - network failures
            logger.error(
                "Failed to fetch Jupiter tokens (attempt %d/3): %s", attempt + 1, exc
            )
        if attempt < 2:
            await asyncio.sleep(0.5 * 2**attempt)

    # Fallback to static registry if Jupiter fails
    if not mapping:
        fetch_url = url or os.getenv("TOKEN_MINTS_URL", TOKEN_REGISTRY_URL)
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(fetch_url, timeout=10) as resp:
                        resp.raise_for_status()
                        data = await resp.json(content_type=None)
                tokens = data.get("tokens") or data.get("data", {}).get("tokens") or []
                temp: Dict[str, str] = {}
                for item in tokens:
                    symbol = item.get("symbol") or item.get("ticker")
                    mint = (
                        item.get("address")
                        or item.get("mint")
                        or item.get("tokenMint")
                    )
                    if isinstance(symbol, str) and isinstance(mint, str):
                        temp[symbol.upper()] = mint
                if temp:
                    mapping = temp
                    break
            except Exception as exc:  # pragma: no cover - network failures
                logger.error(
                    "Failed to fetch token registry (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
            if attempt < 2:
                await asyncio.sleep(0.5 * 2**attempt)

    if not mapping and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                mapping.update({str(k).upper(): str(v) for k, v in cached.items()})
        except Exception as err:  # pragma: no cover - best effort
            logger.error("Failed to read cache: %s", err)

    if not mapping:
        logger.warning("Token mint mapping is empty; cache not written")
        return {}

    TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})
    try:
        from .symbol_utils import invalidate_symbol_cache

        invalidate_symbol_cache()
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(TOKEN_MINTS, f, indent=2)
    except Exception as exc:  # pragma: no cover - optional cache
        logger.error("Failed to write %s: %s", CACHE_FILE, exc)

    _LOADED = True
    return mapping


def _write_cache() -> None:
    """Write ``TOKEN_MINTS`` to :data:`CACHE_FILE`."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(TOKEN_MINTS, f, indent=2)
    except Exception as exc:  # pragma: no cover - optional cache
        logger.error("Failed to write %s: %s", CACHE_FILE, exc)


# Additional mints discovered via manual searches
TOKEN_MINTS.update(
    {
        "AI16Z": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
        "BERA": "A7y2wgyytufsxjg2ub616zqnte3x62f7fcp8fujdmoon",
        "EUROP": "pD6L7wWeei1LJqb7tmnpfEnvkcBvqMgkfqvg23Bpump",
        "FARTCOIN": "Bzc9NZfMqkXR6fz1DBph7BDf9BroyEf6pnzESP7v5iiw",
        "RLUSD": "BkbjmJVa84eiGyp27FTofuQVFLqmKFev4ZPZ3U33pump",
        "USDG": "2gc4f72GkEtggrkUDJRSbLcBpEUPPPFsnDGJJeNKpump",  # Assuming Unlimited Solana Dump
        "VIRTUAL": "2FupRnaRfnyPHg798WsCBMGAauEkrhMs4YN7nBmujPtM",
        "XMR": "Fi9GeixxfhMEGfnAe75nJVrwPqfVefyS6fgmyiTxkS6q",  # Wrapped, verify
        "MELANIA": "FUAfBo2jgks6gB4Z4LfZkqSZgzNucisEHqnNebaRxM1P",
        "PENGU": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
        "USDR": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT as proxy
        "USTC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC proxy (adjust if needed)
        "TRUMP": "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
        # Add more as needed; skip USDQ/USTC/XTZ as non-Solana
    }
)
_write_cache()  # Save immediately


async def refresh_mints() -> None:
    """Force refresh cached token mints and add known symbols."""
    loaded = await load_token_mints(force_refresh=True)
    if not loaded:
        raise RuntimeError("Failed to load token mints")
    TOKEN_MINTS.update(
        {
            "AI16Z": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
            "FARTCOIN": "Bzc9NZfMqkXR6fz1DBph7BDf9BroyEf6pnzESP7v5iiw",
            "MELANIA": "FUAfBo2jgks6gB4Z4LfZkqSZgzNucisEHqnNebaRxM1P",
            "PENGU": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
            "RLUSD": "BkbjmJVa84eiGyp27FTofuQVFLqmKFev4ZPZ3U33pump",
            "VIRTUAL": "2FupRnaRfnyPHg798WsCBMGAauEkrhMs4YN7nBmujPtM",
            "USDG": "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH",
            "USDR": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "USTC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "TRUMP": "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
            "SOL": "So11111111111111111111111111111111111111112",
        }
    )
    _write_cache()
    try:
        from .symbol_utils import invalidate_symbol_cache

        invalidate_symbol_cache()
    except Exception:  # pragma: no cover - best effort
        pass
    logger.info("Refreshed TOKEN_MINTS with %d entries", len(TOKEN_MINTS))


def set_token_mints(mapping: dict[str, str]) -> None:
    """Replace ``TOKEN_MINTS`` with ``mapping`` after normalizing keys."""
    TOKEN_MINTS.clear()
    TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})
    _write_cache()
    try:
        from .symbol_utils import invalidate_symbol_cache

        invalidate_symbol_cache()
    except Exception:  # pragma: no cover - best effort
        pass


async def get_mint_from_gecko(base: str) -> str | None:
    """Return Solana mint address for ``base`` using GeckoTerminal.

    ``None`` is returned if the request fails or no token matches the
    given symbol.
    """

    base_upper = base.upper()
    if base_upper in TOKEN_MINTS:
        return TOKEN_MINTS[base_upper]

    from urllib.parse import quote_plus

    url = (
        "https://api.geckoterminal.com/api/v2/search/pools"
        f"?query={quote_plus(str(base))}&network=solana"
    )

    try:
        data = await gecko_request(url)
    except Exception as exc:
        logger.error("Gecko lookup failed: %s", exc)
        data = None  # Continue to fallback

    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list) and items:
            item = items[0] if isinstance(items[0], dict) else None
            if item:
                mint = (
                    item.get("relationships", {})
                    .get("base_token", {})
                    .get("data", {})
                    .get("id")
                )
                if isinstance(mint, str):
                    if mint.startswith("solana_"):
                        mint = mint[len("solana_") :]
                    return mint

    # Fallback: Helius if Gecko fails
    logger.info("Gecko failed; falling back to Helius for %s", base)
    helius = await fetch_from_helius([base])
    return helius.get(base_upper)


async def fetch_from_helius(symbols: Iterable[str]) -> Dict[str, str]:
    """Return mapping of ``symbols`` to mint addresses using Helius.

    Parameters
    ----------
    symbols:
        Iterable of token symbols to resolve.
    """

    api_key = os.getenv("HELIUS_KEY", "")
    tokens = [str(s) for s in symbols if s]
    if not tokens:
        return {}

    params = {"symbol": ",".join(tokens)}
    if api_key:
        params["api-key"] = api_key
    from urllib.parse import urlencode

    url = f"{HELIUS_TOKEN_API}?{urlencode(params)}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if 400 <= resp.status < 500:
                    text = await resp.text()
                    logger.error("Helius lookup failed [%s]: %s", resp.status, text)
                    return {}
                resp.raise_for_status()
                data = await resp.json()
    except aiohttp.ClientError as exc:  # pragma: no cover - network
        logger.error("Helius lookup failed: %s", exc)
        return {}
    except Exception as exc:  # pragma: no cover - network
        logger.error("Helius lookup error: %s", exc)
        return {}

    result: Dict[str, str] = {}
    if isinstance(data, list):
        items = data
    else:
        items = data.get("tokens") or data.get("data") or []
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("mint") or item.get("address") or item.get("tokenMint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
    return result


async def get_decimals(mint: str) -> int:
    """Return decimal precision for ``mint``.

    The value is first looked up in ``TOKEN_DECIMALS``.  If not found and a
    ``HELIUS_KEY`` is configured, the Helius metadata endpoint is queried and
    the result cached for subsequent calls.
    """

    cached = TOKEN_DECIMALS.get(mint)
    if cached is not None:
        return cached

    api_key = os.getenv("HELIUS_KEY")
    if not api_key:
        return 0

    from urllib.parse import urlencode

    params = {"mint": mint, "api-key": api_key}
    url = f"{HELIUS_TOKEN_API}?{urlencode(params)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Helius decimals lookup failed for %s: %s", mint, exc)
        return 0

    items = data if isinstance(data, list) else data.get("tokens") or data.get("data") or []
    if isinstance(items, dict):
        items = list(items.values())
    for item in items:
        if not isinstance(item, dict):
            continue
        mint_addr = item.get("mint") or item.get("address") or item.get("tokenMint")
        dec = item.get("decimals")
        if isinstance(mint_addr, str) and mint_addr == mint and isinstance(dec, int):
            TOKEN_DECIMALS[mint] = dec
            return dec
    return 0


async def monitor_pump_raydium() -> None:
    """Monitor Pump.fun and Raydium for new tokens."""

    # ``crypto_bot.main`` may not be loaded when this module is imported. Try to
    # access the queue helpers lazily via ``sys.modules`` to avoid triggering an
    # import during test runs or standalone usage.
    import sys

    enqueue_solana_tokens = None  # type: ignore
    _symbol_priority_queue = None  # type: ignore
    main_mod = sys.modules.get("crypto_bot.main")
    if main_mod is not None:  # pragma: no branch - best effort
        enqueue_solana_tokens = getattr(main_mod, "enqueue_solana_tokens", None)
        _symbol_priority_queue = getattr(main_mod, "symbol_priority_queue", None)

    if not TOKEN_MINTS and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                TOKEN_MINTS.update({str(k).upper(): str(v) for k, v in cached.items()})
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Failed to load cache: %s", exc)

    last_pump_ts = datetime.utcnow() - timedelta(minutes=5)
    last_ray_ts = datetime.utcnow() - timedelta(minutes=5)

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                pump_resp, ray_resp = await asyncio.gather(
                    session.get(PUMP_URL, timeout=10),
                    session.get(RAYDIUM_URL, timeout=10),
                )
                pump_data = await pump_resp.json(content_type=None)
                ray_data = await ray_resp.json(content_type=None)

                # Pump.fun tokens
                for item in pump_data if isinstance(pump_data, list) else []:
                    symbol = item.get("symbol")
                    mint = item.get("mint") or item.get("address")
                    created = item.get("created_at") or item.get("createdAt")
                    initial_buy = item.get("initial_buy") or item.get("initialBuy")
                    market_cap = item.get("market_cap") or item.get("marketCap")
                    twitter = item.get("twitter") or item.get("twitter_profile")
                    if not (symbol and mint and created and market_cap):
                        logger.debug(
                            "Skipping Pump.fun token with incomplete data: %s", item
                        )
                        continue
                    try:
                        ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                        if float(market_cap) <= 0:
                            logger.debug(
                                "Skipping Pump.fun token %s due to non-positive market cap %s",
                                symbol,
                                market_cap,
                            )
                            continue
                    except Exception as exc:
                        logger.debug(
                            "Skipping Pump.fun token with invalid data %s: %s", item, exc
                        )
                        continue
                    if ts > last_pump_ts:
                        last_pump_ts = ts
                        key = str(symbol).upper()
                        if key not in TOKEN_MINTS:
                            TOKEN_MINTS[key] = mint
                            logger.info("Pump.fun %s market cap %s", symbol, market_cap)
                            if enqueue_solana_tokens:
                                try:
                                    enqueue_solana_tokens([f"{key}/{mint}"])
                                except Exception as exc:  # pragma: no cover - best effort
                                    logger.error("enqueue_solana_tokens failed: %s", exc)
                            _write_cache()
                            try:
                                from .symbol_utils import invalidate_symbol_cache

                                invalidate_symbol_cache()
                            except Exception:  # pragma: no cover - best effort
                                pass
                            start = ts - timedelta(hours=1)
                            end = ts
                            asyncio.create_task(_fetch_and_train(start, end))

                # Raydium pools
                for pool in ray_data if isinstance(ray_data, list) else []:
                    symbol = (
                        pool.get("baseSymbol")
                        or pool.get("symbol")
                        or pool.get("name")
                    )
                    mint = pool.get("baseMint")
                    created = (
                        pool.get("created_at")
                        or pool.get("createdAt")
                        or pool.get("creationTime")
                    )
                    liquidity = pool.get("liquidity")
                    if not (symbol and mint and created and liquidity):
                        continue
                    try:
                        ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                        if float(liquidity) <= 50_000:
                            continue
                    except Exception:
                        continue
                    if ts > last_ray_ts:
                        last_ray_ts = ts
                        key = str(symbol).split("/")[0].upper()
                        if key not in TOKEN_MINTS:
                            TOKEN_MINTS[key] = mint
                            logger.info("Raydium %s liquidity %s", symbol, liquidity)
                            if enqueue_solana_tokens:
                                try:
                                    enqueue_solana_tokens([f"{key}/{mint}"])
                                except Exception as exc:  # pragma: no cover - best effort
                                    logger.error("enqueue_solana_tokens failed: %s", exc)
                            _write_cache()
                            try:
                                from .symbol_utils import invalidate_symbol_cache

                                invalidate_symbol_cache()
                            except Exception:  # pragma: no cover - best effort
                                pass
                            start = ts - timedelta(hours=1)
                            end = datetime.utcnow()
                            asyncio.create_task(_fetch_and_train(start, end))

                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                logger.info("monitor_pump_raydium cancelled")
                raise
            except Exception as exc:  # pragma: no cover - network errors
                logger.error("monitor_pump_raydium error: %s", exc)
                await asyncio.sleep(10)


# Backward compatibility
monitor_new_tokens = monitor_pump_raydium

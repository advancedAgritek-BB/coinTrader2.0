import asyncio
import time
from threading import Lock
import logging
from typing import AsyncIterator
import contextlib

from .logger import LOG_DIR, setup_logger
try:
    from .symbol_pre_filter import filter_symbols  # legacy dependency
except Exception:
    # Safe default: return the list unchanged when pre-filter module isn't present
    def filter_symbols(symbols):
        return symbols
from .telemetry import telemetry
from .token_registry import TOKEN_MINTS


def fix_symbol(sym: str) -> str:
    """Normalize different notations of Bitcoin."""
    if not isinstance(sym, str):
        return sym
    return sym.replace("XBT/", "BTC/").replace("XBT", "BTC")

logger = setup_logger("bot", LOG_DIR / "bot.log")
pipeline_logger = logging.getLogger("pipeline")


_cached_symbols: tuple[list[tuple[str, float]], list[str]] | None = None
_last_refresh: float = 0.0
_CACHE_INVALIDATION_LOCK = Lock()
_INVALIDATION_GUARD = asyncio.Lock()
_LAST_INVALIDATION_TS = 0.0
_PENDING_INVALIDATION = False
_INVALIDATION_DEBOUNCE_SEC = 300.0


def _apply_invalidation(ts: float | None = None) -> None:
    """Apply the symbol cache invalidation."""
    global _cached_symbols, _last_refresh, _LAST_INVALIDATION_TS, _PENDING_INVALIDATION
    _LAST_INVALIDATION_TS = ts if ts is not None else time.time()
    _cached_symbols = None
    _last_refresh = 0.0
    _PENDING_INVALIDATION = False
    logger.info("Symbol cache invalidated")


def invalidate_symbol_cache() -> None:
    """Clear cached symbols and reset refresh timestamp."""
    global _PENDING_INVALIDATION
    with _CACHE_INVALIDATION_LOCK:
        now = time.time()
        if now - _LAST_INVALIDATION_TS < _INVALIDATION_DEBOUNCE_SEC:
            logger.debug("Symbol cache invalidation suppressed (debounced).")
            return
        if _INVALIDATION_GUARD.locked():
            _PENDING_INVALIDATION = True
            logger.info("Deferring cache invalidation until after evaluation.")
            return
        _apply_invalidation(now)


@contextlib.asynccontextmanager
async def symbol_cache_guard() -> AsyncIterator[None]:
    """Guard operations that should block cache invalidation."""
    await _INVALIDATION_GUARD.acquire()
    try:
        yield
    finally:
        _INVALIDATION_GUARD.release()
        if _PENDING_INVALIDATION:
            invalidate_symbol_cache()


async def get_filtered_symbols(exchange, config) -> tuple[list[tuple[str, float]], list[str]]:
    """Return CEX symbols plus onchain symbols.

    Results are cached for ``symbol_refresh_minutes`` minutes to avoid
    unnecessary API calls.
    """
    global _cached_symbols, _last_refresh

    refresh_m = config.get("symbol_refresh_minutes", 30)
    now = time.time()
    sf = config.get("symbol_filter", {})

    if (
        _cached_symbols is not None
        and now - _last_refresh < refresh_m * 60
    ):
        pipeline_logger.info(
            "discovered_cex=%d discovered_onchain=%d",
            len(_cached_symbols[0]),
            len(_cached_symbols[1]),
        )
        pipeline_logger.info(
            "filtered_liquidity=%d", len(_cached_symbols[0]) + len(_cached_symbols[1])
        )
        return _cached_symbols

    logger.info("Refreshing symbol cache")

    if config.get("skip_symbol_filters"):
        syms = config.get("symbols", [config.get("symbol")])
        result = [(s, 0.0) for s in syms]
        _cached_symbols = (result, [])
        _last_refresh = now
        return result, []

    symbols = config.get("symbols", [config.get("symbol")])
    onchain = list(config.get("onchain_symbols", []))
    pipeline_logger.info(
        "discovered_cex=%d discovered_onchain=%d",
        len(symbols),
        len(onchain),
    )
    if not symbols:
        _cached_symbols = []
        _last_refresh = now
        return [], onchain
    cleaned_symbols = []
    onchain_syms: list[str] = []
    markets = getattr(exchange, "markets", {}) or {}
    for sym in symbols:
        if not isinstance(sym, str):
            cleaned_symbols.append(sym)
            continue
        base, _, quote = sym.partition("/")
        if quote.upper() == "USDC":
            if base.upper() in TOKEN_MINTS:
                onchain_syms.append(sym)
                continue
            if markets is not None and sym in markets:
                cleaned_symbols.append(sym)
                continue
            logger.info("Dropping unsupported USDC pair %s (no mint/market)", sym)
            continue
        cleaned_symbols.append(sym)

    symbols = cleaned_symbols
    skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
    if asyncio.iscoroutinefunction(filter_symbols):
        scored, extra_onchain = await filter_symbols(exchange, symbols, config)
    else:
        scored, extra_onchain = await asyncio.to_thread(
            filter_symbols, exchange, symbols, config
        )
    pipeline_logger.info(
        "filtered_liquidity=%d",
        len(scored) + len(extra_onchain),
    )
    onchain_syms.extend([s for s, _ in extra_onchain])
    skipped_main = telemetry.snapshot().get("scan.symbols_skipped", 0) - skipped_before
    if not scored:
        fallback = config.get("symbol")
        excluded = [s.upper() for s in config.get("excluded_symbols", [])]
        if fallback and fallback.upper() in excluded:
            logger.warning("Fallback symbol %s is excluded", fallback)
            logger.warning(
                "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols",
                skipped_main,
            )
            return [], onchain

        skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
        if asyncio.iscoroutinefunction(filter_symbols):
            check, extra_onchain = await filter_symbols(exchange, [fallback], config)
        else:
            check, extra_onchain = await asyncio.to_thread(
                filter_symbols, exchange, [fallback], config
            )
        onchain_syms.extend([s for s, _ in extra_onchain])
        skipped_fb = telemetry.snapshot().get("scan.symbols_skipped", 0) - skipped_before

        if not check:
            logger.warning(
                "Fallback symbol %s does not meet volume requirements", fallback
            )
            logger.warning(
                "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols initially, %d on fallback",
                skipped_main,
                skipped_fb,
            )
            return [], onchain

        logger.warning(
            "No symbols passed filters, falling back to %s",
            fallback,
        )
        scored = [(fallback, 0.0)]

    logger.info("%d symbols passed filtering", len(scored))

    if not scored:
        logger.warning(
            "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols",
            skipped_main,
        )

    if scored or onchain_syms:
        _cached_symbols = (scored, onchain_syms)
        _last_refresh = now

    exchange_id = getattr(exchange, "id", "unknown")
    quote_whitelist = sf.get("quote_whitelist")
    min_vol = sf.get("min_volume_usd")
    logger.info(
        "Loaded %d markets from %s; %d symbols selected after filters (quote in %s, min_vol=%s)",
        len(markets),
        exchange_id,
        len(scored),
        quote_whitelist,
        min_vol,
    )

    return scored, onchain_syms

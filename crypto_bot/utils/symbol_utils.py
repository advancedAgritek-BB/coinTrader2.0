import asyncio
import time
from threading import Lock
import logging
from typing import AsyncIterator
import contextlib

from .logger import LOG_DIR, setup_logger
from crypto_bot.utils.eval_guard import eval_gate
try:
    from .symbol_pre_filter import filter_symbols  # legacy dependency
except Exception:
    # Safe default: return the list unchanged when pre-filter module isn't present
    def filter_symbols(symbols):
        return symbols
from .telemetry import telemetry
from .token_registry import TOKEN_MINTS
from .market_loader import _is_valid_base_token


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
_LAST_INVALIDATION_TS = 0.0
_INVALIDATION_DEBOUNCE_SEC = 10.0
_AUTO_FALLBACK_WARNED = False
_INVALIDATION_TASK: asyncio.Task | None = None


def _apply_invalidation(ts: float | None = None) -> None:
    """Apply the symbol cache invalidation."""
    global _cached_symbols, _last_refresh, _LAST_INVALIDATION_TS
    _LAST_INVALIDATION_TS = ts if ts is not None else time.time()
    _cached_symbols = None
    _last_refresh = 0.0
    logger.info("Symbol cache invalidated")


def invalidate_symbol_cache() -> None:
    """Clear cached symbols and reset refresh timestamp."""
    global _INVALIDATION_TASK
    with _CACHE_INVALIDATION_LOCK:
        now = time.time()
        if now - _LAST_INVALIDATION_TS < _INVALIDATION_DEBOUNCE_SEC:
            logger.debug("Symbol cache invalidation suppressed (debounced).")
            return
        evaluator = None
        try:
            from crypto_bot.engine.evaluation_engine import get_stream_evaluator

            evaluator = get_stream_evaluator()
        except Exception:
            evaluator = None

        if evaluator is None or not getattr(evaluator, "strategies", None):
            logger.info(
                "No strategies loaded; proceeding with cache invalidation immediately."
            )
            _apply_invalidation(now)
            return

        if eval_gate.is_busy():
            logger.info("Deferring cache invalidation until after evaluation.")
            ttl = getattr(
                getattr(getattr(evaluator, "cfg", None), "evaluation", None),
                "gate_ttl_sec",
                30.0,
            )
            if _INVALIDATION_TASK is None or _INVALIDATION_TASK.done():
                _INVALIDATION_TASK = asyncio.create_task(
                    _deferred_invalidation(now, timeout=ttl)
                )
            return
        _apply_invalidation(now)


@contextlib.asynccontextmanager
async def symbol_cache_guard(note: str = "symbol-cache") -> AsyncIterator[None]:
    """Guard operations that should block cache invalidation."""
    with eval_gate.hold(note):
        yield


async def _wait_for_gate(interval: float) -> None:
    while eval_gate.is_busy():
        await asyncio.sleep(interval)


async def _deferred_invalidation(
    ts: float, timeout: float = 30.0, interval: float = 5.0
) -> None:
    """Retry cache invalidation until the eval gate is free or timeout is reached."""
    try:
        await asyncio.wait_for(_wait_for_gate(interval), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Cache invalidation timed out waiting for evaluation gate.")
        return
    _apply_invalidation(ts)


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

    mode = config.get("mode", "cex")
    sf = config.get("symbol_filter", {})
    allowed_quotes = {
        str(q).upper() for q in (config.get("allowed_quotes") or [])
    }
    markets: dict[str, dict] = {}
    if mode == "cex" and hasattr(exchange, "list_markets"):
        markets = exchange.list_markets()
        if asyncio.iscoroutine(markets):
            markets = await markets
        if isinstance(markets, list):
            markets = {m: {} for m in markets}
        symbols = []
        min_vol = float(sf.get("min_volume_usd", 0) or 0)
        for sym, info in markets.items():
            quote = str(info.get("quote") or sym.split("/")[1]).upper()
            vol = float(info.get("quoteVolume") or info.get("baseVolume") or 0)
            if allowed_quotes and quote not in allowed_quotes:
                continue
            if min_vol and vol < min_vol:
                continue
            symbols.append(sym)
        cfg_syms = config.get("symbols") or [config.get("symbol")]
        if cfg_syms:
            symbols = [s for s in symbols if s in cfg_syms]
        onchain = []
    else:
        symbols = config.get("symbols", [config.get("symbol")])
        onchain = list(config.get("onchain_symbols", []))
        markets = getattr(exchange, "markets", {}) or {}
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
    for sym in symbols:
        if not isinstance(sym, str):
            cleaned_symbols.append(sym)
            continue
        base, _, quote = sym.partition("/")
        if quote.upper() == "USDC":
            if base.upper() in TOKEN_MINTS and mode != "cex":
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

    mode = config.get("mode", "cex")
    cex_candidates = [s for s, _ in scored]
    active_universe = cex_candidates if mode == "cex" else list(onchain_syms)
    if not active_universe:
        if mode == "auto" and cex_candidates:
            global _AUTO_FALLBACK_WARNED
            if not _AUTO_FALLBACK_WARNED:
                logger.warning("Universe empty; falling back to CEX mode")
                _AUTO_FALLBACK_WARNED = True
            config["mode"] = "cex"
        else:
            raise RuntimeError(
                f"Universe is empty (mode={mode}). Check on-chain metadata provider and liquidity filters."
    )

    return scored, onchain_syms


def select_seed_symbols(
    scored: list[tuple[str, float]], exchange, config: dict
) -> list[str]:
    """Return initial evaluation symbols for fast-start mode."""

    fs_cfg = config.get("runtime", {}).get("fast_start", {})
    if not fs_cfg.get("enabled"):
        return [s for s, _ in scored]

    markets = getattr(exchange, "markets", {}) or {}
    seeds_cfg = fs_cfg.get("seed_symbols") or []
    available = [s for s, _ in scored]
    if seeds_cfg:
        seeds = [s for s in seeds_cfg if s in markets and s in available]
    else:
        seed_n = int(fs_cfg.get("seed_batch_size", 15))
        seeds = sorted(
            available,
            key=lambda s: float(markets.get(s, {}).get("quoteVolume") or 0),
            reverse=True,
        )[:seed_n]

    if seeds:
        preview = ", ".join(seeds[:2])
        if len(seeds) > 2:
            preview += ", ..."
        logger.info("Fast-start: seeding %d symbols (%s)", len(seeds), preview)
    return seeds

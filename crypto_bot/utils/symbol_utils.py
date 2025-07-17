import asyncio
import time

from .logger import LOG_DIR, setup_logger
from .symbol_pre_filter import filter_symbols
from .telemetry import telemetry
from .market_loader import _is_valid_base_token


def fix_symbol(sym: str) -> str:
    """Normalize different notations of Bitcoin."""
    if not isinstance(sym, str):
        return sym
    return sym.replace("XBT/", "BTC/").replace("XBT", "BTC")

logger = setup_logger("bot", LOG_DIR / "bot.log")


_cached_symbols: list | None = None
_last_refresh: float = 0.0
_sym_lock = asyncio.Lock()


async def get_filtered_symbols(exchange, config) -> list:
    """Return user symbols filtered by liquidity/volatility or fallback.

    Results are cached for ``symbol_refresh_minutes`` minutes to avoid
    unnecessary API calls.
    """
    global _cached_symbols, _last_refresh

    refresh_m = config.get("symbol_refresh_minutes", 30)
    now = time.time()

    if (
        _cached_symbols is not None
        and now - _last_refresh < refresh_m * 60
    ):
        return _cached_symbols

    symbols = config.get("symbols", [config.get("symbol")])
    filtered_syms: list[str] = []
    for sym in symbols:
        if isinstance(sym, str) and sym.upper().endswith("/USDC"):
            base = sym.split("/", 1)[0]
            if not _is_valid_base_token(base):
                continue
        filtered_syms.append(sym)
    symbols = filtered_syms
    skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
    if asyncio.iscoroutinefunction(filter_symbols):
        scored = await filter_symbols(exchange, symbols, config)
    else:
        scored = await asyncio.to_thread(filter_symbols, exchange, symbols, config)
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
            return []

        skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
        if asyncio.iscoroutinefunction(filter_symbols):
            check = await filter_symbols(exchange, [fallback], config)
        else:
            check = await asyncio.to_thread(filter_symbols, exchange, [fallback], config)
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
            return []

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

    if scored:
        _cached_symbols = scored
        _last_refresh = now

    return scored

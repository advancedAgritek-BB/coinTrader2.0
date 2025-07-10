import asyncio
import time

from .logger import LOG_DIR, setup_logger
from .symbol_pre_filter import filter_symbols
logger = setup_logger(__name__, LOG_DIR / "bot.log")


_cached_symbols: list | None = None
_last_refresh: float = 0.0
_sym_lock = asyncio.Lock()


async def get_filtered_symbols(exchange, config) -> list:
    """Return user symbols filtered by liquidity/volatility or fallback.

    Results are cached for ``symbol_refresh_minutes`` minutes to avoid
    unnecessary API calls.
    """
    global _cached_symbols, _last_refresh

    async with _sym_lock:
        refresh_m = config.get("symbol_refresh_minutes", 30)
        now = time.time()

        if (
            _cached_symbols is not None
            and now - _last_refresh < refresh_m * 60
        ):
            return _cached_symbols

        symbols = config.get("symbols", [config.get("symbol")])
        if asyncio.iscoroutinefunction(filter_symbols):
            scored = await filter_symbols(exchange, symbols, config)
        else:
            scored = await asyncio.to_thread(filter_symbols, exchange, symbols, config)
        if not scored:
            logger.warning(
                "No symbols passed filters, falling back to %s",
                config.get("symbol"),
            )
            scored = [(config.get("symbol"), 0.0)]

        logger.info("%d symbols passed filtering", len(scored))

        _cached_symbols = scored
        _last_refresh = now
        return scored

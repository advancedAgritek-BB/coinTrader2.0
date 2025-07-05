import asyncio
import asyncio
import time

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


_cached_symbols: list | None = None
_last_refresh: float = 0.0


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

    from .symbol_pre_filter import filter_symbols

    symbols = config.get("symbols", [config.get("symbol")])
    if asyncio.iscoroutinefunction(filter_symbols):
        symbols = await filter_symbols(exchange, symbols, config)
    else:
        symbols = await asyncio.to_thread(filter_symbols, exchange, symbols, config)
    if not symbols:
        logger.warning(
            "No symbols passed filters, falling back to %s",
            config.get("symbol"),
        )
        symbols = [config.get("symbol")]

    _cached_symbols = symbols
    _last_refresh = now
    return symbols

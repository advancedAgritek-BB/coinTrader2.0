from .logger import setup_logger
from .symbol_pre_filter import filter_symbols

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


async def get_filtered_symbols(exchange, config) -> list:
    """Return user symbols filtered by liquidity/volatility or fallback."""
    symbols = config.get("symbols", [config.get("symbol")])
    symbols = await filter_symbols(exchange, symbols)
    if not symbols:
        logger.warning(
            "No symbols passed filters, falling back to %s",
            config.get("symbol"),
        )
        symbols = [config.get("symbol")]
    return symbols

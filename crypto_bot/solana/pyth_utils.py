from __future__ import annotations

from typing import Any

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "meme_sniper.log")


async def get_pyth_price(client: Any, symbol: str) -> float:
    """Return the current Pyth price for ``symbol``.

    Parameters
    ----------
    client:
        An initialized :class:`PythClient` instance.
    symbol:
        Symbol to fetch, e.g. ``"SOL/USD"``.
    """
    try:
        from pythclient.pythaccounts import PythPriceType as _PriceType
        await client.refresh_all_prices()
        for product in client.products:
            if getattr(product, "symbol", None) == symbol:
                prices = await product.get_prices()
                price_acc = prices.get(_PriceType.PRICE)
                price = getattr(price_acc, "aggregate_price", None) if price_acc else None
                if price is None:
                    raise ValueError(f"price unavailable for {symbol}")
                return float(price)
        raise ValueError(f"symbol {symbol} not found")
    except Exception as exc:  # pragma: no cover - network issues
        logger.error("Failed to fetch Pyth price: %s", exc)
        raise

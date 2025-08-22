import logging
from typing import Optional
import requests

from crypto_bot.utils.market_loader import get_http_session

logger = logging.getLogger(__name__)


def get_pyth_price(symbol: str) -> Optional[float]:
    """Return latest Pyth price for ``symbol``.

    ``symbol`` should be formatted like ``"BTC/USD"``.
    Returns ``None`` on failure.
    """
    parts = symbol.split("/")
    if len(parts) != 2:
        return None
    base, quote = parts
    query = f"Crypto.{base}/{quote}"
    try:
        session = get_http_session()
        resp = session.get(
            "https://hermes.pyth.network/v2/price_feeds",
            params={"query": query},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        feed_id = data[0].get("id")
        if not feed_id:
            return None
        resp = session.get(
            "https://hermes.pyth.network/api/latest_price_feeds",
            params={"ids[]": feed_id},
            timeout=5,
        )
        resp.raise_for_status()
        price_data = resp.json()[0].get("price")
        if not price_data:
            return None
        price = float(price_data.get("price")) * 10 ** int(price_data.get("expo"))
        return price
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Pyth price fetch failed: %s", exc)
        return None

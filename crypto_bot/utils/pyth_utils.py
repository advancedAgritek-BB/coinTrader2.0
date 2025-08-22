from __future__ import annotations

from typing import Mapping

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "pyth_utils.log")


async def async_get_pyth_price(symbol: str, config: Mapping[str, object]) -> tuple[float, float, bool]:
    """Return the latest Pyth price for ``symbol``.

    Parameters
    ----------
    symbol: str
        Symbol like ``"BTC/USD"``.
    config: Mapping[str, object]
        Configuration with Solana RPC endpoints. Expected keys are
        ``http_endpoint`` and ``ws_endpoint``. Optional ``network``,
        ``program_key`` and ``mapping_key`` override defaults from
        :func:`pythclient.utils.get_key`.

    Returns
    -------
    tuple[float, float, bool]
        ``(price, conf_pct, is_trading)`` where ``conf_pct`` is the
        confidence interval divided by price.
    """
    from pythclient.pythclient import PythClient
    from pythclient.pythaccounts import PythPriceStatus
    from pythclient.utils import get_key

    network = str(config.get("network", "mainnet"))
    http_endpoint = str(config.get("http_endpoint"))
    ws_endpoint = str(config.get("ws_endpoint"))
    program_key = config.get("program_key") or get_key(network, "program")
    mapping_key = config.get("mapping_key") or get_key(network, "mapping")

    logger.info(
        "Fetching Pyth price %s using %s", symbol, http_endpoint
    )

    async with PythClient(
        solana_endpoint=http_endpoint,
        solana_ws_endpoint=ws_endpoint,
        first_mapping_account_key=mapping_key,
        program_key=program_key,
    ) as client:
        await client.refresh_all_prices()
        for product in await client.get_products():
            if product.symbol == symbol:
                price_acc = next(iter(product.prices.values()))
                price = float(price_acc.aggregate_price or 0.0)
                conf = float(price_acc.aggregate_price_confidence_interval or 0.0)
                is_trading = (
                    price_acc.aggregate_price_status == PythPriceStatus.TRADING
                )
                conf_pct = float(conf / price) if price else 0.0
                return price, conf_pct, bool(is_trading)

    raise ValueError(f"symbol {symbol} not found")
import os
from typing import Optional

import requests
from crypto_bot.utils.market_loader import get_http_session


def get_pyth_price(symbol: str, config: Optional[dict] = None) -> float:
    """Return the latest price for ``symbol`` from the Pyth network."""
    mock = os.getenv("MOCK_PYTH_PRICE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0

    # Placeholder: normally we'd query the Pyth API.
    url = config.get("pyth_url") if config else None
    if url:
        try:
            session = get_http_session()
            resp = session.get(f"{url}/{symbol}", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return float(data.get("price", 0.0))
        except Exception:
            pass
    return 0.0

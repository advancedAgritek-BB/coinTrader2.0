"""Exchange helper utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from .kraken_client import KrakenClient, get_kraken_client


@dataclass
class TradeCandidate:
    """Simple structure describing an order to be placed.

    Only the minimal fields required for placing an order are modelled. The
    structure is intentionally lightweight so tests or higher level modules can
    easily construct candidates without depending on a heavy execution layer.
    """

    symbol: str
    side: str
    size: float
    price: Optional[float] = None
    order_type: str = "market"


async def place_order(candidate: TradeCandidate, config: Dict[str, Any]) -> Dict[str, Any]:
    """Place an order for the supplied :class:`TradeCandidate`.

    Parameters
    ----------
    candidate:
        Description of the trade to execute.
    config:
        Runtime configuration. The function honours ``execution_mode`` set to
        ``"dry_run"`` by returning a mock order object. When live trading, an
        exchange client is resolved from ``config`` and used to submit the
        order via ``create_order``.

    Returns
    -------
    dict
        Order information as returned by the exchange. All returned objects are
        guaranteed to contain an ``id`` key for logging.
    """

    mode = str(config.get("execution_mode", "")).lower()
    if mode in {"dry_run", "paper", "paper_trading"} or config.get("paper_trading"):
        # Paper trading – fabricate a lightweight order object
        return {
            "id": f"paper-{uuid4().hex}",
            "symbol": candidate.symbol,
            "side": candidate.side,
            "amount": candidate.size,
            "price": candidate.price,
            "dry_run": True,
        }

    # Resolve or create an exchange client
    client: Any = (
        config.get("exchange_client")
        or config.get("client")
        or config.get("exchange")
    )
    if client is None or isinstance(client, dict):
        ex_cfg = client if isinstance(client, dict) else config.get("exchange", {})
        name = (ex_cfg or {}).get("name", "kraken") if isinstance(ex_cfg, dict) else "kraken"
        params = dict(ex_cfg) if isinstance(ex_cfg, dict) else {}
        params.pop("name", None)
        if name.lower() == "kraken":
            pool_size = config.get("http_pool_size")
            if pool_size is not None:
                params["pool_maxsize"] = int(pool_size)
            client = get_kraken_client(**params)
        else:  # pragma: no cover - optional exchanges
            try:
                import ccxt.async_support as ccxt  # type: ignore
            except Exception:  # pragma: no cover - fallback when async support missing
                import ccxt  # type: ignore
            exchange_cls = getattr(ccxt, name)
            client = exchange_cls(params)

    create = getattr(client, "create_order")
    args = [candidate.symbol, candidate.order_type, candidate.side, candidate.size]
    if candidate.price is not None:
        args.append(candidate.price)

    result = create(*args)
    if asyncio.iscoroutine(result):
        order = await result
    else:
        order = result

    # Ensure an order id is always available for logging
    if isinstance(order, dict):
        order.setdefault("id", order.get("orderId") or order.get("txid") or uuid4().hex)
        return order
    if not hasattr(order, "id"):
        setattr(order, "id", getattr(order, "orderId", uuid4().hex))
    return order  # type: ignore[return-value]


__all__ = ["KrakenClient", "get_kraken_client", "TradeCandidate", "place_order"]

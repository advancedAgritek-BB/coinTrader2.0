import asyncio
import logging
from typing import Any, Dict, Optional

from . import cex_executor

logger = logging.getLogger(__name__)


async def execute_trade_async(
    exchange: Any,
    ws_client: Any,
    symbol: str,
    side: str,
    amount: float,
    notifier: Any,
    *,
    dry_run: bool = True,
    use_websocket: bool = False,
    config: Optional[Dict] = None,
    score: float = 0.0,
    reason: str = "",
    trading_paused: bool = False,
) -> Dict:
    """Wrapper around :mod:`cex_executor` adding detailed logging."""

    if trading_paused:
        logger.info("Trading is paused; signal suppressed (use 'start' to resume)")
        return {"rejection_reason": "trading_paused"}

    if dry_run:
        price = None
        if exchange is not None and hasattr(exchange, "fetch_ticker"):
            try:
                fetch = getattr(exchange, "fetch_ticker")
                if asyncio.iscoroutinefunction(fetch):
                    ticker = await fetch(symbol)
                else:
                    ticker = await asyncio.to_thread(fetch, symbol)
                price = ticker.get("last") or ticker.get("close")
            except Exception:  # pragma: no cover - best effort
                price = None
        logger.info(
            "DRY-RUN: would place %s %.4f %s @ %s; reason=%s",
            side.upper(),
            amount,
            symbol,
            price,
            reason,
        )
        result = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "dry_run": True,
        }
        logger.info(
            "DRY-RUN: simulated %s %.4f %s @ %s",
            side.upper(),
            amount,
            symbol,
            price,
        )
        return result

    def log_event(event: str, order: Dict) -> None:
        logger.info(
            "%s %s %s id=%s score=%s reason=%s",
            event.upper(),
            side.upper(),
            symbol,
            order.get("id"),
            score,
            reason,
        )

    result = await cex_executor.execute_trade_async(
        exchange,
        ws_client,
        symbol,
        side,
        amount,
        notifier,
        dry_run=False,
        use_websocket=use_websocket,
        config=config,
        score=score,
        event_cb=log_event,
    )

    price = result.get("price") if isinstance(result, dict) else None
    if not result:
        logger.warning(
            "ORDER FAILED: %s %.4f %s @ %s",
            side.upper(),
            amount,
            symbol,
            price,
        )
    elif result.get("rejection_reason"):
        logger.warning(
            "ORDER REJECTED: %s %.4f %s @ %s reason=%s",
            side.upper(),
            amount,
            symbol,
            price,
            result.get("rejection_reason"),
        )
    else:
        logger.info(
            "ORDER EXECUTED: %s %.4f %s @ %s",
            side.upper(),
            amount,
            symbol,
            price,
        )

    return result

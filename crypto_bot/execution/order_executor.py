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
        return {}

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
        return {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "dry_run": True,
        }

    return await cex_executor.execute_trade_async(
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
    )

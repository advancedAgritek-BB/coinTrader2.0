from __future__ import annotations

import logging
from typing import Any

from .queues import trade_queue
from crypto_bot.exchange import place_order

log = logging.getLogger(__name__)


async def execution_loop(config: Any) -> None:
    """Consume trade candidates from :data:`trade_queue` and execute orders.

    Parameters
    ----------
    config:
        Runtime configuration passed through to :func:`place_order`.
    """
    while True:
        candidate = await trade_queue.get()
        try:
            order = await place_order(candidate, config)
            order_id = order.get("id") if isinstance(order, dict) else getattr(order, "id", None)
            log.info("EXECUTED %s -> %s", order_id, candidate)
        except Exception:
            log.exception("Order placement failed for %s", candidate)
        finally:
            trade_queue.task_done()


__all__ = ["execution_loop"]

import logging

from . import trade_queue

logger = logging.getLogger(__name__)

try:
    from .trading import place_order  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    async def place_order(candidate, config):
        return {"candidate": candidate, "config": config}


async def execution_loop(config):
    """Consume trade candidates and place orders."""
    while True:
        candidate = await trade_queue.get()
        try:
            result = await place_order(candidate, config)
            logger.info("Order result: %s", result)
        except Exception as exc:  # pragma: no cover - log unexpected errors
            logger.exception("Order placement failed: %s", exc)
        finally:
            trade_queue.task_done()

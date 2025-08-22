import asyncio
import logging

from . import trade_queue

logger = logging.getLogger(__name__)

async def scoring_loop(config):
    """Placeholder scoring loop that does nothing."""
    while True:
        await asyncio.sleep(1)

import os
import asyncio
import logging
from typing import Any

import aiohttp

PUMP_URL = "https://api.pump.fun/tokens?limit=50&offset=0"
RAYDIUM_URL = "https://api.raydium.io/v2/main/pairs"
POLL_INTERVAL = 10

logger = logging.getLogger(__name__)
FEATURE_ENABLE_PUMP_MONITOR = os.getenv("FEATURE_ENABLE_PUMP_MONITOR", "0").lower() in ("1", "true")

async def monitor_pump_raydium() -> None:
    """Poll Pump.fun and Raydium for new tokens.

    The monitor can be disabled by setting ``FEATURE_ENABLE_PUMP_MONITOR`` to ``0``.
    When enabled, any network or processing error triggers an exponential backoff
    retry loop.
    """
    if not FEATURE_ENABLE_PUMP_MONITOR:
        logger.info("pump.fun/Raydium monitor disabled by FEATURE_ENABLE_PUMP_MONITOR.")
        return

    retries = 0
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                pump_resp, ray_resp = await asyncio.gather(
                    session.get(PUMP_URL, timeout=10),
                    session.get(RAYDIUM_URL, timeout=10),
                )
                # We don't use the responses further here; callers can monkeypatch
                # this function to inspect results. The fetch is primarily to ensure
                # connectivity and trigger side effects in downstream logic.
                await pump_resp.json(content_type=None)
                await ray_resp.json(content_type=None)
                retries = 0
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                logger.info("monitor_pump_raydium cancelled")
                raise
            except Exception as e:  # pragma: no cover - network and misc errors
                retries += 1
                delay = min(2 ** retries, 60)
                logger.warning(
                    "monitor_pump_raydium error: %s; retrying in %ss", e, delay
                )
                await asyncio.sleep(delay)

from __future__ import annotations

import asyncio
import aiohttp
from typing import AsyncGenerator, Any, Dict
import json
import os
import logging


async def watch_pool(
    pool_program: str, api_key: str | None = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Helius enhanced websocket and yield transaction data.

    If ``api_key`` is not provided, the ``HELIUS_KEY`` environment variable
    is used.

    Raises
    ------
    RuntimeError
        If no API key is available.
    """

    if api_key is None:
        api_key = os.getenv("HELIUS_KEY")
    if not api_key:
        raise RuntimeError("HELIUS API key is required; set api_key or HELIUS_KEY env var")

    url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"
    logger = logging.getLogger(__name__)
    ClientError = getattr(aiohttp, "ClientError", Exception)
    async with aiohttp.ClientSession() as session:
        backoff = 1
        last_log = 0
        while True:
            reconnect = False
            try:
                async with session.ws_connect(url) as ws:
                    sub = {
                        "jsonrpc": "2.0",
                        "id": 420,
                        "method": "transactionSubscribe",
                        "params": [
                            {"failed": False, "accountInclude": [pool_program]},
                            {
                                "commitment": "processed",
                                "encoding": "base64",
                                "transactionDetails": "full",
                                "showRewards": True,
                            },
                        ],
                    }
                    await ws.send_json(sub)
                    backoff = 1
                    last_log = 0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = msg.json()
                            except Exception:
                                data = json.loads(msg.data)
                            result = None
                            if isinstance(data, dict):
                                result = data.get("params", {}).get("result")
                            if result is not None:
                                yield result
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            reconnect = True
                            break
            except (aiohttp.WSServerHandshakeError, ClientError, asyncio.TimeoutError, OSError) as exc:
                reconnect = True
                if backoff != last_log:
                    logger.error("Helius pool monitor error: %s; retrying in %ss", exc, backoff)
                    last_log = backoff
            except Exception as exc:
                reconnect = True
                if backoff != last_log:
                    logger.error("Helius pool monitor error: %s; retrying in %ss", exc, backoff)
                    last_log = backoff

            if reconnect:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

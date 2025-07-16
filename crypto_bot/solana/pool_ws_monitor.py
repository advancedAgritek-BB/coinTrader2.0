from __future__ import annotations

import aiohttp
from typing import AsyncGenerator, Any, Dict
import json
import logging

logger = logging.getLogger(__name__)

async def watch_pool(api_key: str, pool_program: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Helius enhanced websocket and yield transaction data."""
    url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"
    async with aiohttp.ClientSession() as session:
        try:
            ws = await session.ws_connect(url)
        except aiohttp.WSServerHandshakeError as e:
            logger.error(
                "WebSocket connection failed. Your API key may be invalid or lacks access: %s",
                e,
            )
            return
        async with ws:
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
                    break

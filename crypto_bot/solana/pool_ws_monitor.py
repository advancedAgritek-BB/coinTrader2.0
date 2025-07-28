from __future__ import annotations

import asyncio
import aiohttp
from typing import AsyncGenerator, Any, Dict
import json


async def watch_pool(
    api_key: str, pool_program: str, min_liquidity: float = 0.0
) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Helius enhanced websocket and yield transaction data."""
    url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"
    async with aiohttp.ClientSession() as session:
        backoff = 0
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
                    backoff = 0
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
                                liquidity = float(result.get("liquidity", 0.0))
                                tx_count = int(result.get("txCount", result.get("tx_count", 0)))
                                if liquidity < min_liquidity or tx_count <= 10:
                                    continue
                                yield result
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            reconnect = True
                            break
            except aiohttp.WSServerHandshakeError:
                reconnect = True
            except Exception:
                reconnect = True

            if reconnect:
                backoff += 1
                await asyncio.sleep(2 ** backoff)

from __future__ import annotations

"""WebSocket monitor for Solana liquidity pools.

Run as a module to print pool events from Helius:

```
HELIUS_KEY=your_key python -m crypto_bot.solana.pool_ws_monitor [-p PROGRAM_ID]
```

The program ID defaults to Raydium's AMM if not supplied via the
``PROGRAM_ID`` environment variable or ``-p`` argument.
"""

import asyncio
import argparse
import json
import os
from typing import Any, AsyncGenerator, Dict

import aiohttp

DEFAULT_PROGRAM_ID = "EhhTK0i58FmSPrbr30Y8wVDDDeWGPAHDq6vNru6wUATk"


async def watch_pool(
    api_key: str, pool_program: str, min_liquidity: float = 0.0
) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Helius enhanced websocket and yield transaction data."""
    # Use the standard Helius mainnet endpoint by default. Enhanced endpoints
    # such as ``atlas-mainnet.helius-rpc.com`` are only available to business
    # plans, so point to the broadly accessible URL.
    url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
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
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            reconnect = True
                            break
                    else:
                        # Loop exhausted without break -> no more messages
                        return
            except aiohttp.WSServerHandshakeError:
                reconnect = True
            except Exception:
                reconnect = True

            if reconnect:
                backoff += 1
                await asyncio.sleep(2 ** backoff)
            else:
                return


async def main(args: list[str] | None = None) -> None:
    """Print pool events using :func:`watch_pool`."""
    parser = argparse.ArgumentParser(description="Stream Solana pool events")
    parser.add_argument(
        "-p",
        "--program",
        default=os.getenv("PROGRAM_ID", DEFAULT_PROGRAM_ID),
        help="Program ID to monitor (default: env PROGRAM_ID or Raydium)",
    )
    parser.add_argument(
        "-k",
        "--key",
        dest="api_key",
        default=os.getenv("HELIUS_KEY", ""),
        help="Helius API key (default: HELIUS_KEY)",
    )
    opts = parser.parse_args(args)

    async for event in watch_pool(opts.api_key, opts.program):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())

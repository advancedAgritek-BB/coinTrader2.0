from __future__ import annotations

"""WebSocket monitor for Solana liquidity pools.

Run as a module to print pool events from Helius:

```
python -m crypto_bot.solana.pool_ws_monitor -k YOUR_KEY [-p PROGRAM_ID]
```

Both the API key and program ID may also be supplied via ``HELIUS_KEY`` and
``PROGRAM_ID`` environment variables. The program defaults to Raydium's AMM
if not provided.
"""

import asyncio
import argparse
import json
import os
from typing import Any, AsyncGenerator, Dict

import aiohttp
import logging
import numpy as np

DEFAULT_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

logger = logging.getLogger(__name__)


def parse_liquidity(tx_data: dict) -> float:
    """Compare pre/post token balances and return absolute amount change."""
    meta = tx_data.get("meta") or {}
    pre = meta.get("preTokenBalances") or []
    post = meta.get("postTokenBalances") or []

    def _to_dict(balances: list[dict]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for bal in balances:
            mint = bal.get("mint")
            amount = bal.get("uiTokenAmount", {}).get("uiAmount")
            try:
                result[mint] = float(amount)
            except (TypeError, ValueError):
                continue
        return result

    pre_map = _to_dict(pre)
    post_map = _to_dict(post)

    total = 0.0
    for mint, post_amt in post_map.items():
        pre_amt = pre_map.pop(mint, 0.0)
        total += abs(post_amt - pre_amt)
    for remaining in pre_map.values():
        total += abs(remaining)
    return total


def predict_regime(tx_data: dict) -> str:
    """Predict regime label using a trained model."""
    try:
        from coinTrader_Trainer.ml_trainer import load_model  # type: ignore

        model = load_model("regime_lgbm")
        features = np.array([[parse_liquidity(tx_data), tx_data.get("tx_count", 0)]])
        labels = ["trending", "volatile", "breakout", "mean-reverting"]
        try:
            pred = (
                model.predict_proba(features)[0]
                if hasattr(model, "predict_proba")
                else model.predict(features)[0]
            )
        except Exception:
            pred = model.predict(features)[0]
        idx = int(np.argmax(pred)) if hasattr(pred, "__len__") else int(pred)
        return labels[idx]
    except Exception:
        logger.exception("predict_regime failed")
        return "unknown"


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
                                "encoding": "jsonParsed",
                                "transactionDetails": "full",
                                "showRewards": True,
                                "maxSupportedTransactionVersion": 0,
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
                                if data.get("method") != "transactionNotification":
                                    continue
                                result = data.get("params", {}).get("result")
                            if result is not None:
                                liquidity = parse_liquidity(result)
                                tx_count = result.get("tx_count", 0)
                                result["tx_count"] = tx_count
                                if liquidity >= min_liquidity:
                                    result["predicted_regime"] = predict_regime(result)
                                    logger.info("Regime predicted: %s", result["predicted_regime"])
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

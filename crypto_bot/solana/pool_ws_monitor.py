"""WebSocket monitor for Solana liquidity pools.

Run as a module to print pool events from Helius::

    python -m crypto_bot.solana.pool_ws_monitor -k YOUR_KEY [-p PROGRAM_ID]

Both the API key and program ID may also be supplied via ``HELIUS_KEY`` and
``PROGRAM_ID`` environment variables. The program defaults to Raydium's AMM if
not provided.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict

import aiohttp
import numpy as np

DEFAULT_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

logger = logging.getLogger(__name__)


def _amount(balance: Dict[str, Any]) -> float:
    """Return uiAmount from a token balance entry."""

    try:
        return float(balance.get("uiTokenAmount", {}).get("uiAmount", 0.0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def parse_liquidity(tx_data: Dict[str, Any]) -> float:
    """Compare pre/post token balances and return the total absolute change."""

    meta = tx_data.get("meta") or {}
    pre = meta.get("preTokenBalances") or []
    post = meta.get("postTokenBalances") or []

    total = 0.0
    for a, b in zip(pre, post):
        total += abs(_amount(a) - _amount(b))
    for extra in pre[len(post):]:
        total += abs(_amount(extra))
    for extra in post[len(pre):]:
        total += abs(_amount(extra))
    return total


def predict_regime(tx_data: Dict[str, Any]) -> str:
    """Predict regime label using a trained model.

    Falls back to ``"breakout"`` if the model is unavailable.
    """

    labels = ["trending", "volatile", "breakout", "mean-reverting"]
    try:  # pragma: no cover - best effort
        from coinTrader_Trainer.ml_trainer import load_model  # type: ignore

        model = load_model("regime_lgbm")
        features = np.array(
            [[parse_liquidity(tx_data), tx_data.get("tx_count", tx_data.get("txCount", 0))]]
        )
        probs = (
            model.predict_proba(features)[0]
            if hasattr(model, "predict_proba")
            else model.predict(features)[0]
        )
        idx = int(np.argmax(probs)) if hasattr(probs, "__len__") else int(probs)
        return labels[idx]
    except Exception:  # pragma: no cover - missing optional dependency
        logger.exception("predict_regime failed")
        return "breakout"


async def watch_pool(
    api_key: str, pool_program: str, min_liquidity: float = 0.0
) -> AsyncGenerator[Dict[str, Any], None]:
    """Connect to Helius websocket and yield qualifying transactions."""

    url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
    backoff = 0
    async with aiohttp.ClientSession() as session:
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
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            if msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                reconnect = True
                                break
                            continue
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
                                regime = predict_regime(result)
                                if regime == "breakout":
                                    result["predicted_regime"] = regime
                                    logger.info("Regime predicted: %s", regime)
                                    yield result
                    else:
                        # Loop exhausted without break -> no more messages
                        return
            except aiohttp.WSServerHandshakeError:
                reconnect = True
            except Exception:  # pragma: no cover - reconnection path
                logger.exception("watch_pool reconnecting")
                reconnect = True

            if reconnect:
                backoff = min(backoff + 1, 5)
                await asyncio.sleep(2 ** backoff)
                continue
            break


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


if __name__ == "__main__":  # pragma: no cover - manual invocation
    asyncio.run(main())


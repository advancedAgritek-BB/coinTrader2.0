"""Pool watcher utilities for Solana."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
import os

import aiohttp
import logging
import yaml
import pandas as pd


@dataclass
class NewPoolEvent:
    """Event emitted when a new liquidity pool is detected."""

    pool_address: str
    token_mint: str
    creator: str
    liquidity: float
    tx_count: int = 0
    freeze_authority: str = ""
    mint_authority: str = ""
    timestamp: float = 0.0


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


logger = logging.getLogger(__name__)


class PoolWatcher:
    """Async engine that polls for new pools and yields :class:`NewPoolEvent`."""
    def __init__(
        self,
        url: str | None = None,
        interval: float | None = None,
        websocket_url: str | None = None,
        raydium_program_id: str | None = None,
        max_failures: int = 3,
        min_liquidity: float | None = None,
        ml_filter: bool | None = None,
    ) -> None:
        pump_fun_program_id = None
        if (
            url is None
            or interval is None
            or websocket_url is None
            or raydium_program_id is None
            or min_liquidity is None
        ):
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            sniper_cfg = cfg.get("meme_wave_sniper", {})
            pool_cfg = sniper_cfg.get("pool", {})
            safety_cfg = sniper_cfg.get("safety", {})
            pump_fun_program_id = pool_cfg.get("pump_fun_program_id")
            if url is None:
                url = pool_cfg.get("url", "")
            if interval is None:
                interval = float(pool_cfg.get("interval", 5))
            if websocket_url is None:
                websocket_url = pool_cfg.get("websocket_url", "")
            if raydium_program_id is None:
                raydium_program_id = pool_cfg.get("raydium_program_id", "")
            if min_liquidity is None:
                min_liquidity = float(safety_cfg.get("min_liquidity", 0))
            if ml_filter is None:
                ml_filter = bool(pool_cfg.get("ml_filter", False))
        key = os.getenv("HELIUS_KEY")
        if not url or "YOUR_KEY" in url or url.endswith("api-key="):
            if not key:
                raise ValueError(
                    "Helius API key missing. Set HELIUS_KEY or update pool.url"
                )
            if not url:
                url = f"https://mainnet.helius-rpc.com/v1/?api-key={key}"
            else:
                url = url.replace("YOUR_KEY", key)
                if url.endswith("api-key="):
                    url += key
        if websocket_url:
            if "${HELIUS_KEY}" in websocket_url:
                websocket_url = websocket_url.replace("${HELIUS_KEY}", key or "")
            if "YOUR_KEY" in websocket_url:
                websocket_url = websocket_url.replace("YOUR_KEY", key or "")
        self.url = url
        self.interval = interval
        self.websocket_url = websocket_url
        self.raydium_program_id = raydium_program_id
        self.program_ids = [self.raydium_program_id]
        if pump_fun_program_id:
            self.program_ids.append(pump_fun_program_id)
        self.min_liquidity = float(min_liquidity or 0)
        self.ml_filter = bool(ml_filter)
        self._running = False
        self._seen: set[str] = set()
        self._max_failures = max_failures
        self._failures = 0

    def _predict_breakout(self, event: NewPoolEvent) -> float:
        """Return breakout probability using the Supabase snapshot."""
        if not self.ml_filter or not event.token_mint:
            return 1.0
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return 1.0
        try:  # pragma: no cover - optional dependency
            from supabase import create_client
            from coinTrader_Trainer import regime_lgbm

            client = create_client(url, key)
            data = client.table("snapshots").select("*").eq("token_mint", event.token_mint).single().execute()
            snapshot = getattr(data, "data", data)
            return float(regime_lgbm.predict(snapshot))
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("ML filter failed: %s", exc)
            return 1.0

    async def watch(self) -> AsyncGenerator[NewPoolEvent, None]:
        """Yield NewPoolEvent from WebSocket subscription."""
        self._running = True
        backoff = 0
        async with aiohttp.ClientSession() as session:
            while self._running:
                reconnect = False
                try:
                    async with session.ws_connect(self.websocket_url) as ws:
                        for prog_id in self.program_ids:
                            request = {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "transactionSubscribe",
                                "params": [
                                    {"failed": False, "accountInclude": [prog_id]},
                                    {
                                        "commitment": "processed",
                                        "encoding": "jsonParsed",
                                        "transactionDetails": "full",
                                        "maxSupportedTransactionVersion": 0,
                                    },
                                ],
                            }
                            await ws.send_json(request)
                        backoff = 0
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                reconnect = True
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            try:
                                data = json.loads(msg.data)
                                if data.get("method") != "transactionNotification":
                                    continue
                                result = data.get("params", {}).get("result", {})
                                logs = result.get("transaction", {}).get("meta", {}).get("logMessages", [])
                                account_keys = (
                                    result.get("transaction", {})
                                    .get("transaction", {})
                                    .get("message", {})
                                    .get("accountKeys", [])
                                )

                                if any("initialize2: InitializeInstruction2" in log for log in logs):
                                    pool_addr = account_keys[2] if len(account_keys) > 2 else ""
                                    token_mint = account_keys[1] if len(account_keys) > 1 else ""
                                    creator = account_keys[0] if account_keys else ""
                                    event = NewPoolEvent(pool_address=pool_addr, token_mint=token_mint, creator=creator, liquidity=0.0)
                                    if self._predict_breakout(event) >= 0.5:
                                        yield event

                                elif any("Program log: Instruction: InitializeMint2" in log for log in logs):
                                    token_mint = account_keys[1] if len(account_keys) > 1 else ""
                                    creator = account_keys[0] if account_keys else ""
                                    event = NewPoolEvent(pool_address="", token_mint=token_mint, creator=creator, liquidity=0.0)
                                    if self._predict_breakout(event) >= 0.5:
                                        yield event

                            except Exception as e:
                                logger.error("Event parsing error: %s", e)
                except Exception as e:
                    reconnect = True
                    logger.error("WS connection error: %s", e)

                if self._running and reconnect:
                    backoff = min(backoff + 1, 5)
                    await asyncio.sleep(2 ** backoff)



    def stop(self) -> None:
        """Stop the watcher loop."""
        self._running = False

    async def _fetch_snapshot(self, pool_addr: str) -> pd.DataFrame:
        """Return a simple OHLCV-like snapshot for ``pool_addr``."""
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [pool_addr, {"limit": 20}],
                },
            ) as resp:
                data = await resp.json()
        result = data.get("result", [])
        rows = [
            {"timestamp": r.get("blockTime", 0), "volume": 1}
            for r in result
        ]
        return pd.DataFrame(rows)

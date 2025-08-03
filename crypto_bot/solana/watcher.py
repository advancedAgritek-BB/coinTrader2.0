from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
import os
import time

import aiohttp
import logging
import yaml
import requests  # For webhook setup


@dataclass
class NewPoolEvent:
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
    def __init__(
        self,
        url: str | None = None,
        interval: float | None = None,
        websocket_url: str | None = None,
        raydium_program_id: str | None = None,
        min_liquidity: float | None = None,
        ml_filter: bool | None = None,
    ) -> None:
        if (
            url is None
            or interval is None
            or websocket_url is None
            or raydium_program_id is None
            or ml_filter is None
        ):
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            sniper_cfg = cfg.get("meme_wave_sniper", {})
            pool_cfg = sniper_cfg.get("pool", {})
            safety_cfg = sniper_cfg.get("safety", {})
            if url is None:
                url = pool_cfg.get("url", "")
            if interval is None:
                interval = float(pool_cfg.get("interval", 5))
            if websocket_url is None:
                websocket_url = pool_cfg.get("websocket_url", "")
            if raydium_program_id is None:
                raydium_program_id = pool_cfg.get(
                    "raydium_program_id",
                    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
                )  # Updated to v4
            if ml_filter is None:
                ml_filter = bool(pool_cfg.get("ml_filter", False))
        if min_liquidity is None:
            min_liquidity = 0.0
        key = os.getenv("HELIUS_KEY")
        if not key:
            raise ValueError("HELIUS_KEY not set")
        if not url:
            url = f"https://api.helius.xyz/v0/?api-key={key}"
        self.url = url
        self.interval = interval
        self.websocket_url = websocket_url or f"wss://mainnet.helius-rpc.com/?api-key={key}"
        self.raydium_program_id = raydium_program_id
        self.program_ids = [self.raydium_program_id]
        self.min_liquidity = float(min_liquidity or 0)
        self.ml_filter = bool(ml_filter)
        self._running = False
        self._seen: set[str] = set()

        # One-time webhook setup
        self.setup_webhook(key)

    def setup_webhook(self, api_key: str):
        """Register a webhook with Helius if configured."""
        webhook_url = os.getenv("BOT_WEBHOOK_URL", "")
        if not webhook_url:
            logger.info("Webhook disabled; BOT_WEBHOOK_URL not set")
            return

        helius_url = f"https://api.helius.xyz/v0/webhooks?api-key={api_key}"
        payload = {
            "webhookURL": webhook_url,
            "transactionTypes": ["CREATE_POOL", "ADD_LIQUIDITY", "TOKEN_MINT"],
            "accountAddresses": [self.raydium_program_id],
            "webhookType": "enhanced",
        }
        try:
            response = requests.post(helius_url, json=payload)
            if response.status_code == 200:
                logger.info("Webhook setup successful")
            else:
                logger.error(f"Webhook setup failed: {response.text}")
        except Exception as e:  # pragma: no cover - best effort
            logger.error(f"Webhook error: {e}")

    def _predict_breakout(self, event: NewPoolEvent) -> float:
        """Predict breakout using regime_lgbm (from coinTrader_Trainer via Supabase)."""
        if not self.ml_filter or not event.token_mint:
            return 1.0
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            logger.warning("Supabase creds missing; skipping ML")
            return 1.0
        try:  # pragma: no cover - optional dependency
            from supabase import create_client
            from coinTrader_Trainer.ml_trainer import load_model

            client = create_client(supabase_url, supabase_key)
            data = (
                client.table("snapshots")
                .select("*")
                .eq("token_mint", event.token_mint)
                .execute()
            )
            snapshot = data.data[0] if data.data else None
            if snapshot:
                model = load_model("regime_lgbm")
                features = [
                    snapshot.get("liquidity", 0),
                    snapshot.get("volume", 0),
                    event.tx_count,
                ]
                try:
                    pred = (
                        model.predict_proba([features])[0]
                        if hasattr(model, "predict_proba")
                        else model.predict([features])[0]
                    )
                except Exception:  # pragma: no cover - best effort
                    pred = model.predict([features])[0]

                if hasattr(pred, "__iter__"):
                    return float(pred[1])
                return float(pred == 1)
            return 0.0
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(f"ML prediction failed: {exc}")
            return 0.0

    async def watch(self) -> AsyncGenerator[NewPoolEvent, None]:
        self._running = True
        backoff = 0
        async with aiohttp.ClientSession() as session:
            while self._running:
                reconnect = False
                try:
                    async with session.ws_connect(self.websocket_url) as ws:
                        for i, prog_id in enumerate(self.program_ids):
                            req = {
                                "jsonrpc": "2.0",
                                "id": i,
                                "method": "transactionSubscribe",
                                "params": [
                                    {"failed": False, "accountInclude": [prog_id]},
                                    {
                                        "commitment": "processed",
                                        "encoding": "jsonParsed",
                                        "transactionDetails": "full",
                                        "showRewards": True,
                                        "maxSupportedTransactionVersion": 0,
                                    },
                                ],
                            }
                            await ws.send_json(req)
                        backoff = 0
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                reconnect = True
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            try:
                                data = json.loads(msg.data)
                                if data.get("method") != "transactionNotification":
                                    continue
                                result = data.get("params", {}).get("result", {})
                                signature = result.get("signature", "")
                                if not signature or signature in self._seen:
                                    continue
                                self._seen.add(signature)
                                tx = result.get("transaction", {})
                                if tx and "initialize" in json.dumps(tx).lower():
                                    event = NewPoolEvent("", "", "", 0.0)
                                    await self._enrich_event(event, signature, session)
                                    if event.liquidity >= self.min_liquidity and self._predict_breakout(event) >= 0.5:
                                        yield event
                                        logger.info(f"New pool event yielded: {event}")
                            except Exception as exc:  # pragma: no cover - best effort
                                logger.error(f"Event parsing error: {exc}")
                except Exception as exc:  # pragma: no cover - best effort
                    reconnect = True
                    logger.error(f"WS connection error: {exc}")

                if self._running and reconnect:
                    backoff = min(backoff + 1, 5)
                    await asyncio.sleep(2 ** backoff)

    async def _enrich_event(
        self,
        event: NewPoolEvent,
        signature: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """Enrich event with full tx details (from Helius docs)."""
        try:
            async with session.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTransaction",
                    "params": [
                        signature,
                        {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
                    ],
                },
            ) as resp:
                data = await resp.json()
                result = data.get("result", {})
                meta = result.get("meta", {})
                message = result.get("transaction", {}).get("message", {})

                keys = []
                for k in message.get("accountKeys", []):
                    if isinstance(k, dict):
                        keys.append(k.get("pubkey", ""))
                    else:
                        keys.append(str(k))
                if len(keys) > 2:
                    event.pool_address = keys[2]
                if len(keys) > 1:
                    event.token_mint = keys[1]
                if keys:
                    event.creator = keys[0]

                pre = meta.get("preTokenBalances", [])
                post = meta.get("postTokenBalances", [])
                liquidity = 0.0
                tx_count = 0
                for p, q in zip(pre, post):
                    try:
                        pre_amt = float(q.get("uiTokenAmount", {}).get("uiAmount", 0))
                        post_amt = float(p.get("uiTokenAmount", {}).get("uiAmount", 0))
                        diff = abs(post_amt - pre_amt)
                        if diff > 0:
                            liquidity += diff
                            tx_count += 1
                    except Exception:
                        continue
                event.liquidity = liquidity
                event.tx_count = tx_count
                event.timestamp = float(result.get("blockTime") or time.time())
                logger.debug(
                    f"Enriched event: liquidity={liquidity}, tx_count={tx_count}"
                )
        except Exception as exc:  # pragma: no cover - best effort
            logger.error(f"Enrich error: {exc}")

    def stop(self) -> None:
        self._running = False

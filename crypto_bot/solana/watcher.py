"""Pool watcher utilities for Solana."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional
import os

import aiohttp
import logging
import yaml


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
    ) -> None:
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
        self.min_liquidity = float(min_liquidity or 0)
        self._running = False
        self._seen: set[str] = set()
        self._max_failures = max_failures
        self._failures = 0

    async def watch(self) -> AsyncGenerator[NewPoolEvent, None]:
        """Yield :class:`NewPoolEvent` objects as they are discovered."""
        if self.websocket_url and self.raydium_program_id:
            async for evt in self._watch_ws():
                yield evt
            return

        self._running = True
        ClientError = getattr(aiohttp, "ClientError", Exception)
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.post(
                        self.url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "dex.getNewPools",
                            "params": {"protocols": ["raydium"], "limit": 50},
                        },
                        timeout=10,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self._failures = 0
                except aiohttp.ClientResponseError as e:
                    if e.status == 404:
                        self._failures += 1
                        logger.error(
                            "PoolWatcher error: 404 from %s - the configured URL is invalid or no longer supported. "
                            "Try https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY",
                            self.url,
                        )
                        if self._failures >= self._max_failures:
                            self._running = False
                            raise RuntimeError(f"Pools endpoint not found: {self.url}") from e
                        await asyncio.sleep(self.interval)
                        continue
                    logger.error("PoolWatcher error: %s", e)
                    await asyncio.sleep(self.interval)
                    continue
                except (ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.error("PoolWatcher error: %s", e)
                    await asyncio.sleep(self.interval)
                    continue

                result = data.get("result", data)
                pools = result.get("pools") or result.get("data") or result
                for item in pools:
                    addr = (
                        item.get("address")
                        or item.get("poolAddress")
                        or item.get("pool_address")
                        or ""
                    )
                    if not addr or addr in self._seen:
                        continue
                    liquidity = float(item.get("liquidity", 0.0))
                    if liquidity < self.min_liquidity:
                        continue
                    self._seen.add(addr)
                    yield NewPoolEvent(
                        pool_address=addr,
                        token_mint=item.get("tokenMint")
                        or item.get("token_mint")
                        or "",
                        creator=item.get("creator", ""),
                        liquidity=liquidity,
                        tx_count=int(item.get("txCount", item.get("tx_count", 0))),
                        freeze_authority=item.get("freezeAuthority")
                        or item.get("freeze_authority")
                        or "",
                        mint_authority=item.get("mintAuthority")
                        or item.get("mint_authority")
                        or "",
                        timestamp=float(item.get("timestamp", 0.0)),
                    )

                await asyncio.sleep(self.interval)

    async def _watch_ws(self) -> AsyncGenerator[NewPoolEvent, None]:
        """Yield events from a websocket subscription."""
        self._running = True
        backoff = 1
        last_log = 0
        ClientError = getattr(aiohttp, "ClientError", Exception)
        async with aiohttp.ClientSession() as session:
            while self._running:
                reconnect = False
                try:
                    async with session.ws_connect(self.websocket_url) as ws:
                        await ws.send_json(
                            {
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "programSubscribe",
                                "params": [self.raydium_program_id, {"encoding": "jsonParsed"}],
                            }
                        )
                        backoff = 1
                        last_log = 0
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
                            except ValueError:
                                continue
                            if data.get("method") != "programNotification":
                                continue
                            value = (
                                data.get("params", {})
                                .get("result", {})
                                .get("value", {})
                            )
                            addr = value.get("pubkey") or ""
                            if not addr or addr in self._seen:
                                continue
                            self._seen.add(addr)
                            yield NewPoolEvent(
                                pool_address=addr,
                                token_mint="",
                                creator="",
                                liquidity=0.0,
                            )
                except aiohttp.WSServerHandshakeError as e:
                    reconnect = True
                    if e.status == 401:
                        logger.error(
                            "Unauthorized WebSocket connection – check HELIUS_KEY or subscription tier"
                        )
                        self._running = False
                        return
                    if backoff != last_log:
                        logger.error(
                            "WebSocket handshake failed with status %s; retrying in %ss",
                            e.status,
                            backoff,
                        )
                        last_log = backoff
                except (ClientError, asyncio.TimeoutError, OSError) as e:
                    reconnect = True
                    if backoff != last_log:
                        logger.error("WS error: %s; retrying in %ss", e, backoff)
                        last_log = backoff
                except Exception as e:  # pragma: no cover - generic catch for stability
                    reconnect = True
                    if backoff != last_log:
                        logger.error("WS error: %s; retrying in %ss", e, backoff)
                        last_log = backoff

                if self._running and reconnect:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)


    def stop(self) -> None:
        """Stop the watcher loop."""
        self._running = False

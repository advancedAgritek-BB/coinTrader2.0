"""Pool watcher utilities for Solana."""

from __future__ import annotations

import asyncio
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

    def __init__(self, url: str | None = None, interval: float | None = None) -> None:
        if url is None or interval is None:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            sniper_cfg = cfg.get("meme_wave_sniper", {})
            pool_cfg = sniper_cfg.get("pool", {})
            if url is None:
                url = pool_cfg.get("url", "")
            if interval is None:
                interval = float(pool_cfg.get("interval", 5))
        key = os.getenv("HELIUS_KEY")
        if not url or "YOUR_KEY" in url or url.endswith("api-key="):
            if not key:
                raise ValueError(
                    "Helius API key missing. Set HELIUS_KEY or update pool.url"
                )
            if not url:
                url = f"https://api.helius.xyz/v0/pools?api-key={key}"
            else:
                url = url.replace("YOUR_KEY", key)
                if url.endswith("api-key="):
                    url += key
        self.url = url
        self.interval = interval
        self._running = False
        self._seen: set[str] = set()

    async def watch(self) -> AsyncGenerator[NewPoolEvent, None]:
        """Yield :class:`NewPoolEvent` objects as they are discovered."""
        self._running = True
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.get(self.url, timeout=10) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.error("PoolWatcher error: %s", e)
                    await asyncio.sleep(self.interval)
                    continue

                pools = data.get("pools") or data.get("data") or data
                for item in pools:
                    addr = (
                        item.get("address")
                        or item.get("poolAddress")
                        or item.get("pool_address")
                        or ""
                    )
                    if not addr or addr in self._seen:
                        continue
                    self._seen.add(addr)
                    yield NewPoolEvent(
                        pool_address=addr,
                        token_mint=item.get("tokenMint")
                        or item.get("token_mint")
                        or "",
                        creator=item.get("creator", ""),
                        liquidity=float(item.get("liquidity", 0.0)),
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

    def stop(self) -> None:
        """Stop the watcher loop."""
        self._running = False

from __future__ import annotations

"""Scheduler that captures 1m candles for newly listed tokens."""

import asyncio
import json
import time
from pathlib import Path
from typing import Mapping

import aiohttp

from .watcher import NewPoolEvent
from crypto_bot.utils.logger import LOG_DIR


class CandleScheduler:
    """Capture 1m candles for tokens in the background."""

    def __init__(self, url_template: str, minutes: int = 15) -> None:
        self.url_template = url_template
        self.minutes = minutes
        self.tasks: dict[str, asyncio.Task] = {}

    def schedule(self, event: NewPoolEvent) -> None:
        """Start capturing candles for ``event`` if not already running."""
        if event.token_mint and event.token_mint not in self.tasks:
            self.tasks[event.token_mint] = asyncio.create_task(
                self._collect(event.token_mint)
            )

    async def close(self) -> None:
        """Cancel all active capture tasks."""
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()

    async def _collect(self, token_mint: str) -> None:
        end = time.time() + self.minutes * 60
        path = LOG_DIR / f"{token_mint}_1m.jsonl"
        async with aiohttp.ClientSession() as session:
            while time.time() < end:
                try:
                    async with session.get(
                        self.url_template.format(mint=token_mint), timeout=10
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "a") as f:
                        json.dump(data, f)
                        f.write("\n")
                except Exception:
                    await asyncio.sleep(60)
                    continue
                await asyncio.sleep(60)

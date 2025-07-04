from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional


async def run(
    exchange: object,
    paper_wallet: Optional[object] = None,
    log_file: str | Path = "crypto_bot/logs/bot.log",
) -> None:
    """Periodically output balance and last log line.

    This coroutine runs until cancelled and is intentionally lightweight so
    tests can easily patch it. The monitor fetches the current balance from
    ``exchange`` or ``paper_wallet`` and prints the last line of ``log_file``.
    """
    log_path = Path(log_file)
    last_line = ""
    while True:
        await asyncio.sleep(5)
        balance = None
        try:
            if paper_wallet is not None:
                balance = getattr(paper_wallet, "balance", None)
            elif hasattr(exchange, "fetch_balance"):
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance")):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                balance = bal.get("USDT", {}).get("free", 0) if isinstance(bal.get("USDT"), dict) else bal.get("USDT", 0)
        except Exception:
            pass
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            if lines:
                last_line = lines[-1]
        print(f"[Monitor] balance={balance} log='{last_line}'")

from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional


async def monitor_loop(
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
    prev_len = 0
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
            for line in reversed(lines):
                if "Loading config" not in line:
                    last_line = line
                    break

        message = f"[Monitor] balance={balance} log='{last_line}'"
        print(" " * prev_len, end="\r")
        print(message, end="\r", flush=True)
        prev_len = len(message)
"""Simple console monitor for displaying trades."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import log_reader

TRADE_FILE = Path("crypto_bot/logs/trades.csv")


def display_trades(
    exchange: Any | None = None, wallet: Any | None = None, trade_file: Path = TRADE_FILE
) -> str:
    """Read trades from ``trade_file`` and print them as a table.

    Returns the rendered table as text so tests can verify the output.
    """
    df = log_reader._read_trades(trade_file)
    console = Console(record=True)
    table = Table(show_header=True, header_style="bold")
    table.add_column("symbol")
    table.add_column("side")
    table.add_column("amount")
    table.add_column("price")

    for _, row in df.iterrows():
        table.add_row(
            str(row.get("symbol", "")),
            str(row.get("side", "")),
            str(row.get("amount", "")),
            str(row.get("price", "")),
        )

    console.print(table)
    return console.export_text()

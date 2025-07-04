from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional, Any


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
    prev_first = 0
    prev_second = 0
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
        stats = await trade_stats_line(exchange)

        print(" " * prev_second, end="\r")
        print("\033[F" + " " * prev_first, end="\r")

        output = message
        if stats:
            output += "\n" + stats
        print(output, end="\r", flush=True)

        prev_first = len(message)
        prev_second = len(stats) if stats else 0
"""Simple console monitor for displaying trades."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .utils.open_trades import get_open_trades

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


async def trade_stats_line(exchange: Any, trade_file: Path = TRADE_FILE) -> str:
    """Return a single line summarizing PnL for each open trade."""
    open_trades = get_open_trades(trade_file)
    if not open_trades:
        return ""

    symbols = {t["symbol"] for t in open_trades}
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                ticker = await exchange.fetch_ticker(sym)
            else:
                ticker = await asyncio.to_thread(exchange.fetch_ticker, sym)
            prices[sym] = float(ticker.get("last") or ticker.get("close") or 0.0)
        except Exception:
            prices[sym] = 0.0

    parts = []
    for trade in open_trades:
        sym = trade.get("symbol")
        entry = float(trade.get("price", 0))
        amount = float(trade.get("amount", 0))
        pnl = (prices.get(sym, 0.0) - entry) * amount
        parts.append(f"{sym} {pnl:+.2f}")
    return " | ".join(parts)

from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional, Any

from crypto_bot.utils.logger import LOG_DIR
import sys

TRADE_FILE = LOG_DIR / "trades.csv"

from rich.console import Console
from rich.table import Table

from .utils.open_trades import get_open_trades
from . import log_reader



async def monitor_loop(
    exchange: object,
    paper_wallet: Optional[object] = None,
    log_file: str | Path = LOG_DIR / "bot.log",
    trade_file: str | Path = TRADE_FILE,
) -> None:
    """Periodically output balance, last log line and open trade stats.

    This coroutine runs until cancelled and is intentionally lightweight so
    tests can easily patch it. The monitor fetches the current balance from
    ``exchange`` or ``paper_wallet`` and prints the last line of ``log_file``.
    Open trade PnL lines are generated from ``trade_file`` and printed below the
    status line when positions exist.
    """
    log_path = Path(log_file)
    last_line = ""
    prev_lines = 0
    prev_output = ""
    offset = 0

    try:
        with log_path.open("r", encoding="utf-8") as fh:
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

                fh.seek(offset)
                for line in fh:
                    if "Loading config" not in line:
                        last_line = line.rstrip("\n")
                offset = fh.tell()

                message = f"[Monitor] balance={balance} log='{last_line}'"
                lines = await trade_stats_lines(exchange, Path(trade_file))

                output = message
                if lines:
                    output += "\n" + "\n".join(lines)

                if sys.stdout.isatty():
                    # Clear previously printed lines
                    if prev_lines:
                        print("\033[2K", end="")
                        for _ in range(prev_lines - 1):
                            print("\033[F\033[2K", end="")
                    print(output, end="\r", flush=True)
                    prev_lines = output.count("\n") + 1
                    prev_output = output
                else:
                    if output != prev_output:
                        print(output)
                        prev_output = output
    except asyncio.CancelledError:
        # Propagate cancellation after the file handle is closed by the
        # context manager.
        raise


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


async def trade_stats_lines(exchange: Any, trade_file: Path = TRADE_FILE) -> list[str]:
    """Return a list of lines summarizing PnL for each open trade."""
    open_trades = get_open_trades(trade_file)
    if not open_trades:
        return []

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

    lines = []
    for trade in open_trades:
        sym = trade.get("symbol")
        entry = float(trade.get("price", 0))
        amount = float(trade.get("amount", 0))
        side = trade.get("side", "long")
        current = prices.get(sym, 0.0)
        if side == "short":
            pnl = (entry - current) * amount
        else:
            pnl = (current - entry) * amount
        lines.append(f"{sym} -- {entry:.2f} -- {pnl:+.2f}")
    return lines


async def trade_stats_line(exchange: Any, trade_file: Path = TRADE_FILE) -> str:
    """Return a single line summarizing PnL for each open trade."""
    lines = await trade_stats_lines(exchange, trade_file)
    return " | ".join(lines)

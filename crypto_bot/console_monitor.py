from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional, Any

from crypto_bot.utils.logger import LOG_DIR
import sys
import csv
import pandas as pd

# Flag indicating whether ``monitor_loop`` is currently running. The main
# application checks this to disable its own status output when the console
# monitor is active.
MONITOR_ACTIVE = False

TRADE_FILE = LOG_DIR / "trades.csv"

from rich.console import Console
from rich.table import Table

from . import log_reader


def get_open_trades(log_path: Path | str) -> list[dict]:
    """Return open positions parsed from ``log_path``."""
    path = Path(log_path)
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 6 and str(row[5]).strip().lower() == "true":
                continue
            if len(row) < 5:
                row = row + [None] * (5 - len(row))
            rows.append(row[:5])

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["symbol", "side", "amount", "price", "timestamp"])

    long_positions: dict[str, list[dict]] = {}
    short_positions: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        try:
            qty = float(row.get("amount", 0))
        except Exception:
            qty = 0.0
        try:
            price = float(row.get("price", 0))
        except Exception:
            price = 0.0
        entry_time = row.get("timestamp")

        if side == "buy":
            shorts = short_positions.get(symbol, [])
            while qty > 0 and shorts:
                pos = shorts[0]
                if qty >= pos["amount"]:
                    qty -= pos["amount"]
                    shorts.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if shorts:
                short_positions[symbol] = shorts
            else:
                short_positions.pop(symbol, None)
            if qty > 0:
                long_positions.setdefault(symbol, []).append(
                    {
                        "symbol": symbol,
                        "side": "long",
                        "amount": qty,
                        "price": price,
                        "entry_time": entry_time,
                    }
                )
        elif side == "sell":
            longs = long_positions.get(symbol, [])
            while qty > 0 and longs:
                pos = longs[0]
                if qty >= pos["amount"]:
                    qty -= pos["amount"]
                    longs.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if longs:
                long_positions[symbol] = longs
            else:
                long_positions.pop(symbol, None)
            if qty > 0:
                short_positions.setdefault(symbol, []).append(
                    {
                        "symbol": symbol,
                        "side": "short",
                        "amount": qty,
                        "price": price,
                        "entry_time": entry_time,
                    }
                )

    result: list[dict] = []
    for positions in long_positions.values():
        result.extend(positions)
    for positions in short_positions.values():
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result



async def monitor_loop(
    exchange: object,
    paper_wallet: Optional[object] = None,
    log_file: str | Path = LOG_DIR / "bot.log",
    trade_file: str | Path = TRADE_FILE,
    *,
    quiet_mode: bool = False,
) -> None:
    """Periodically print the latest log line, balance and open trade stats.

    This coroutine runs until cancelled and is intentionally lightweight so
    tests can easily patch it. The monitor fetches the current balance from
    ``exchange`` or ``paper_wallet`` and prints the most recent line of
    ``log_file``. A separate balance line and one line per open trade PnL are
    rendered below. Set ``quiet_mode`` to ``True`` to print a single update when
    stdout is not a TTY.
    """
    global MONITOR_ACTIVE
    MONITOR_ACTIVE = True
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
                        balance = (
                            bal.get("USDT", {}).get("free", 0)
                            if isinstance(bal.get("USDT"), dict)
                            else bal.get("USDT", 0)
                        )
                except Exception:
                    pass

                fh.seek(offset)
                for line in fh:
                    if "Loading config" not in line:
                        last_line = line.rstrip("\n")
                offset = fh.tell()

                log_line = last_line
                balance_line = f"Balance: {balance}"
                trade_lines = await trade_stats_lines(exchange, Path(trade_file))

                output = "\n".join([log_line, balance_line, *trade_lines])

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
                    if quiet_mode:
                        if not prev_output:
                            print(output)
                            prev_output = output
                    elif output != prev_output:
                        print(output)
                        prev_output = output
    except asyncio.CancelledError:
        # Propagate cancellation after the file handle is closed by the
        # context manager.
        raise
    finally:
        MONITOR_ACTIVE = False


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
    missing = set(symbols)

    if getattr(exchange, "has", {}).get("fetchTickers"):
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_tickers", None)):
                tickers = await exchange.fetch_tickers(list(symbols))
            else:
                tickers = await asyncio.to_thread(exchange.fetch_tickers, list(symbols))
            for sym, ticker in (tickers or {}).items():
                try:
                    prices[sym] = float(ticker.get("last") or ticker.get("close") or 0.0)
                except Exception:
                    prices[sym] = 0.0
                missing.discard(sym)
        except Exception:
            pass

    for sym in missing:
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

from __future__ import annotations

import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import csv

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


LOG_FILE = Path("crypto_bot/logs/trades.csv")
console = Console()


def get_open_trades(path: Path = LOG_FILE) -> List[Dict[str, float | str]]:
    """Return a list of open trades from the trades CSV."""
    if not path.exists():
        return []
    trades = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "symbol":
                continue
            symbol, side, amount, price, *_ = row
            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "amount": float(amount),
                    "price": float(price),
                }
            )
    open_positions: List[Dict[str, float | str]] = []
    for t in trades:
        if t["side"] == "buy":
            open_positions.append(
                {
                    "symbol": t["symbol"],
                    "side": "buy",
                    "amount": t["amount"],
                    "entry_price": t["price"],
                }
            )
        else:
            qty = t["amount"]
            i = 0
            while qty > 0 and i < len(open_positions):
                pos = open_positions[i]
                if pos["symbol"] != t["symbol"]:
                    i += 1
                    continue
                traded = min(pos["amount"], qty)
                pos["amount"] -= traded
                if pos["amount"] == 0:
                    open_positions.pop(i)
                    i -= 1
                qty -= traded
                i += 1
    return [p for p in open_positions if p.get("amount", 0) > 0]


async def _fetch_balance(exchange, paper_wallet: Optional[object]) -> float:
    if paper_wallet is not None:
        return float(getattr(paper_wallet, "balance", 0.0))
    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
        bal = await exchange.fetch_balance()
    else:
        bal = await asyncio.to_thread(exchange.fetch_balance)
    usdt = bal.get("USDT")
    return float(usdt.get("free", usdt) if isinstance(usdt, dict) else usdt or 0.0)


async def _fetch_prices(exchange, symbols: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym in symbols:
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                t = await exchange.fetch_ticker(sym)
            else:
                t = await asyncio.to_thread(exchange.fetch_ticker, sym)
            price = t.get("last") or t.get("close") or 0.0
            prices[sym] = float(price)
        except Exception:
            prices[sym] = 0.0
    return prices


async def run(exchange, paper_wallet: Optional[object] = None) -> None:
    """Continuously display wallet balance and open trades."""
    table = Table(show_header=False)
    with Live(table, console=console, refresh_per_second=1):
        while True:
            balance = await _fetch_balance(exchange, paper_wallet)
            trades = get_open_trades()
            symbols = list({t["symbol"] for t in trades})
            prices = await _fetch_prices(exchange, symbols) if symbols else {}

            table = Table(show_header=False)
            table.add_row(f"Balance: {balance:.2f} USDT")
            table.add_row(f"Active trades: {len(trades)}")
            for t in trades:
                price = prices.get(t["symbol"], 0.0)
                pnl = (price - t.get("entry_price", 0)) * t.get("amount", 0)
                if t.get("side") == "sell":
                    pnl *= -1
                color = "green" if pnl >= 0 else "red"
                pnl_text = Text(f"{pnl:.2f}", style=color)
                table.add_row(f"{t['symbol']} {t['side']} {t['amount']}", pnl_text)

            console.clear()
            console.print(table)
            await asyncio.sleep(60)


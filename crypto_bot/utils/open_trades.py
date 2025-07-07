from __future__ import annotations

"""Utility for detecting currently open trades from the trade log."""

from pathlib import Path
from typing import Dict, List

from crypto_bot import log_reader



def get_open_trades(log_path: Path) -> List[Dict]:
    """Return a list of open trade entries from ``log_path``.

    Each returned dictionary contains ``symbol``, ``side``, ``amount``, ``price``
    and ``entry_time`` keys. Buy orders are matched with later sells on a FIFO
    basis to determine remaining open quantity per symbol.
    """
    df = log_reader._read_trades(log_path)
    if df.empty:
        return []

    # Map symbol to list of open positions in order of entry
    open_positions: Dict[str, List[Dict]] = {}

    for _, row in df.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        try:
            amount = float(row.get("amount", 0))
        except Exception:
            amount = 0.0
        try:
            price = float(row.get("price", 0))
        except Exception:
            price = 0.0
        entry_time = row.get("timestamp")

        if side == "buy":
            open_positions.setdefault(symbol, []).append(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "amount": amount,
                    "price": price,
                    "entry_time": entry_time,
                }
            )
            continue

        if side == "sell":
            qty = amount
            positions = open_positions.get(symbol, [])
            while qty > 0 and positions:
                pos = positions[0]
                if pos["amount"] <= qty:
                    qty -= pos["amount"]
                    positions.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if not positions:
                open_positions.pop(symbol, None)

    # Flatten remaining positions preserving entry order
    result: List[Dict] = []
    for positions in open_positions.values():
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result

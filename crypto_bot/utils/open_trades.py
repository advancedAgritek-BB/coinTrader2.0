from __future__ import annotations

"""Utility for detecting currently open trades from the trade log.

The previous implementation only tracked remaining long positions created by
``buy`` orders.  This version keeps independent FIFO queues for both long and
short entries so that unmatched ``sell`` orders create short positions and
subsequent ``buy`` orders can offset them.
"""

from pathlib import Path
from typing import Dict, List

from crypto_bot import log_reader



def get_open_trades(log_path: Path) -> List[Dict]:
    """Return a list of open trade entries from ``log_path``.

    The returned dictionaries contain ``symbol``, ``side`` (``"long"`` or
    ``"short"``), ``amount``, ``price`` and ``entry_time`` keys. Buy orders are
    matched with sells on a FIFO basis while unmatched sells produce short
    entries that future buys may offset.
    """
    df = log_reader._read_trades(log_path)
    if df.empty:
        return []

    # Maintain separate FIFO queues for long and short entries per symbol
    long_positions: Dict[str, List[Dict]] = {}
    short_positions: Dict[str, List[Dict]] = {}

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
            qty = amount
            shorts = short_positions.get(symbol, [])
            # Close existing shorts first
            while qty > 0 and shorts:
                pos = shorts[0]
                if pos["amount"] <= qty:
                    qty -= pos["amount"]
                    shorts.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if not shorts:
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
            continue

        if side == "sell":
            qty = amount
            longs = long_positions.get(symbol, [])
            # Close existing longs first
            while qty > 0 and longs:
                pos = longs[0]
                if pos["amount"] <= qty:
                    qty -= pos["amount"]
                    longs.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if not longs:
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

    # Flatten remaining positions preserving entry order
    result: List[Dict] = []
    for positions in long_positions.values():
        result.extend(positions)
    for positions in short_positions.values():
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result

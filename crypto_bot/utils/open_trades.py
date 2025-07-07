from __future__ import annotations

"""Utility for detecting currently open trades from the trade log.

Earlier versions only tracked long entries created by ``buy`` orders. This
module keeps separate FIFO queues for long and short entries so unmatched
``sell`` orders become short positions and later ``buy`` orders can offset
them.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from crypto_bot import log_reader



def get_open_trades(log_path: Path) -> List[Dict]:
    """Return remaining open trades from ``log_path``.

    Each returned dictionary contains ``symbol``, ``side`` (``"long"`` or
    ``"short"``), ``amount``, ``price`` and ``entry_time`` keys. Buy orders
    offset existing short positions before adding new long ones while sell
    orders close longs first.
    The returned dictionaries contain ``symbol``, ``side`` (``"long"`` or
    ``"short"``), ``amount``, ``price`` and ``entry_time`` keys. Buy orders are
    matched with sells on a FIFO basis while unmatched sells produce short
    entries that future buys may offset.
    Each result dictionary includes ``symbol``, ``side`` (``"long"`` or
    ``"short"``), ``amount``, ``price`` and ``entry_time``. Buy orders are
    matched with later sells on a FIFO basis and excess sells create short
    positions.
    """
    df = log_reader._read_trades(log_path)
    if df.empty:
        return []

    open_longs: Dict[str, List[Dict]] = {}
    open_shorts: Dict[str, List[Dict]] = {}
    # Maintain separate FIFO queues for long and short entries per symbol
    long_positions: Dict[str, List[Dict]] = {}
    short_positions: Dict[str, List[Dict]] = {}

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
            qty = amount
            shorts = open_shorts.get(symbol, [])
            shorts = short_positions.get(symbol, [])
            # Offset existing shorts first
            while qty > 0 and shorts:
                pos = shorts[0]
                if qty >= pos["amount"]:
                    qty -= pos["amount"]
                    shorts.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if not shorts:
                open_shorts.pop(symbol, None)
            if qty > 0:
                open_longs.setdefault(symbol, []).append(
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
            longs = open_longs.get(symbol, [])
        elif side == "sell":
            longs = long_positions.get(symbol, [])
            # Offset existing longs first
            while qty > 0 and longs:
                pos = longs[0]
                if qty >= pos["amount"]:
                    qty -= pos["amount"]
                    longs.pop(0)
                else:
                    pos["amount"] -= qty
                    qty = 0
            if not longs:
                open_longs.pop(symbol, None)
            if qty > 0:
                open_shorts.setdefault(symbol, []).append(
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

    result: List[Dict] = []
    for positions in open_longs.values():
        result.extend(positions)
    for positions in open_shorts.values():
    for positions in long_positions.values():
        result.extend(positions)
    for positions in short_positions.values():
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result

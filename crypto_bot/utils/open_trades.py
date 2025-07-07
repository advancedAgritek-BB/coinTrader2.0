from __future__ import annotations

"""Utility for detecting currently open trades from the trade log."""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from crypto_bot import log_reader



def get_open_trades(log_path: Path) -> List[Dict]:
    """Return a list of open trade entries from ``log_path``.

    Each returned dictionary contains ``symbol``, ``side`` (``"long"`` or
    ``"short"``), ``amount``, ``price`` and ``entry_time`` keys. Buy orders
    offset existing short positions before adding new long ones while sell
    orders close longs first.
    """
    df = log_reader._read_trades(log_path)
    if df.empty:
        return []

    open_longs: Dict[str, List[Dict]] = {}
    open_shorts: Dict[str, List[Dict]] = {}

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
            shorts = open_shorts.get(symbol, [])
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
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result

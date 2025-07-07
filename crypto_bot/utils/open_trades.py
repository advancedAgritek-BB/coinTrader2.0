from __future__ import annotations

"""Utility for computing currently open trades from the trade log."""

from pathlib import Path
from typing import Dict, List

from crypto_bot import log_reader


def get_open_trades(log_path: Path | str) -> List[Dict]:
    """Return a list of open trades remaining in ``log_path``.

    Each result dictionary contains ``symbol``, ``side`` (``"long"`` or ``"short"``),
    ``amount``, ``price`` and ``entry_time`` keys. Buy orders close existing short
    positions before opening longs while sell orders close longs first. Any
    remaining quantity becomes a new position which is tracked on a FIFO basis.
    """
    df = log_reader._read_trades(log_path)
    if df.empty:
        return []

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

    result: List[Dict] = []
    for positions in long_positions.values():
        result.extend(positions)
    for positions in short_positions.values():
        result.extend(positions)

    result.sort(key=lambda x: x.get("entry_time"))
    return result

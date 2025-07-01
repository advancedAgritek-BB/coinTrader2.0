from __future__ import annotations

"""Simple trade logger for tax purposes."""

from datetime import datetime
from typing import Dict, List
import pandas as pd

# store open positions and closed trades
_open_positions: List[Dict] = []
_closed_trades: List[Dict] = []


def _extract_token(symbol: str) -> str:
    if symbol and '/' in symbol:
        return symbol.split('/')[0]
    return symbol


def _extract_price(order: Dict) -> float:
    for key in ("price", "avg_price", "average", "fill_price", "cost"):
        if key in order and order[key]:
            value = order[key]
            if key == "cost" and order.get("amount"):
                return float(value) / float(order["amount"])
            return float(value)
    return float(order.get("amount", 0)) * 0.0


def record_entry(order: Dict) -> None:
    """Record a purchase with timestamp."""
    token = _extract_token(order.get("symbol", order.get("token_in", "")))
    _open_positions.append(
        {
            "time": datetime.utcnow(),
            "token": token,
            "qty": float(order.get("amount", 0)),
            "price": _extract_price(order),
        }
    )


def record_exit(order: Dict) -> None:
    """Record a sale, computing PnL and holding period."""
    token = _extract_token(order.get("symbol", order.get("token_out", "")))
    if not _open_positions:
        return
    entry = _open_positions.pop(0)
    sell_price = _extract_price(order)
    pnl = (sell_price - entry["price"]) * entry["qty"]
    days = (datetime.utcnow() - entry["time"]).days
    trade_type = "long_term" if days >= 365 else "short_term"
    _closed_trades.append(
        {
            "Date": datetime.utcnow().date().isoformat(),
            "Token": token or entry["token"],
            "Qty": entry["qty"],
            "Buy_Price": entry["price"],
            "Sell_Price": sell_price,
            "Profit": pnl,
            "Type": trade_type,
        }
    )


def export_csv(path: str) -> None:
    """Export closed trades to CSV at ``path``."""
    if not _closed_trades:
        df = pd.DataFrame(columns=[
            "Date",
            "Token",
            "Qty",
            "Buy_Price",
            "Sell_Price",
            "Profit",
            "Type",
        ])
    else:
        df = pd.DataFrame(_closed_trades)
    df.to_csv(path, index=False)

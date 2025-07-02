from __future__ import annotations

"""Utility for computing trade statistics from log files."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _read_trades(path: Path | str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["symbol", "side", "amount", "price"])
    df = pd.read_csv(file, header=None)
    cols = ["symbol", "side", "amount", "price", "timestamp"]
    df = df.iloc[:, : len(cols)]
    df.columns = cols[: df.shape[1]]
    return df


def trade_summary(path: Path | str) -> Dict[str, float]:
    df = _read_trades(path)
    num_trades = len(df)
    pnl = 0.0
    wins = 0
    closed = 0
    open_positions: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        side = row.get("side")
        price = float(row.get("price", 0))
        amount = float(row.get("amount", 0))
        if side == "buy":
            open_positions.append((price, amount))
        elif side == "sell" and open_positions:
            entry_price, qty = open_positions.pop(0)
            traded = min(qty, amount)
            profit = (price - entry_price) * traded
            pnl += profit
            closed += 1
            if profit > 0:
                wins += 1
            if qty > traded:
                open_positions.insert(0, (entry_price, qty - traded))
    win_rate = wins / closed if closed else 0.0
    active = sum(qty for _, qty in open_positions)
    return {
        "num_trades": num_trades,
        "total_pnl": pnl,
        "win_rate": win_rate,
        "active_positions": active,
    }

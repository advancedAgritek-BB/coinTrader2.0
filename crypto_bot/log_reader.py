from __future__ import annotations

"""Utility for computing trade statistics from log files."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _read_trades(path: Path | str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["symbol", "side", "amount", "price", "timestamp"])

    cols = ["symbol", "side", "amount", "price", "timestamp"]
    df = pd.read_csv(
        file,
        header=None,
        names=cols,
        engine="python",
        on_bad_lines=lambda row: row[: len(cols)],
    )
    df = df.iloc[:, : len(cols)]
    return df


def trade_summary(path: Path | str) -> Dict[str, float]:
    df = _read_trades(path)
    num_trades = len(df)
    pnl = 0.0
    wins = 0
    closed = 0
    # Track open long and short trades separately
    open_longs: List[Tuple[float, float]] = []
    open_shorts: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        side = row.get("side")
        try:
            price = float(row.get("price", 0))
        except Exception:
            price = 0.0
        try:
            amount = float(row.get("amount", 0))
        except Exception:
            amount = 0.0
        if side == "buy":
            # Buy orders first close any open shorts
            if open_shorts:
                entry_price, qty = open_shorts.pop(0)
                traded = min(qty, amount)
                profit = (entry_price - price) * traded
                pnl += profit
                closed += 1
                if profit > 0:
                    wins += 1
                if qty > traded:
                    open_shorts.insert(0, (entry_price, qty - traded))
                amount -= traded
            # Remaining amount opens new long position
            if amount > 0:
                open_longs.append((price, amount))
        elif side == "sell":
            # Sell orders close longs first
            if open_longs:
                entry_price, qty = open_longs.pop(0)
                traded = min(qty, amount)
                profit = (price - entry_price) * traded
                pnl += profit
                closed += 1
                if profit > 0:
                    wins += 1
                if qty > traded:
                    open_longs.insert(0, (entry_price, qty - traded))
                amount -= traded
            # Excess sell amount opens short position
            if amount > 0:
                open_shorts.append((price, amount))
    # When no round-trip trades have been closed, assume a neutral bootstrap
    # win rate of ``0.6`` so downstream components are not overly penalised by
    # lack of history.
    win_rate = wins / closed if closed else 0.6
    active = sum(qty for _, qty in open_longs) + sum(qty for _, qty in open_shorts)
    return {
        "num_trades": num_trades,
        "total_pnl": pnl,
        "win_rate": win_rate,
        "active_positions": active,
    }

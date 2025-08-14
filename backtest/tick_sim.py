"""Simple tick-level backtesting simulator.

This module provides a command line interface for simulating very
basic maker/taker strategies on pre-recorded order book snapshots.  It
is intentionally lightweight and serves as a quick way to experiment
with microstructure ideas before deploying them live.

Snapshots are expected to be stored as a JSON lines (``.jsonl``) file.
Each line should follow a minimal schema with the following fields::

    {
        "ts": 1700000000.0,     # Unix timestamp in seconds
        "bids": [[price, qty], ...],
        "asks": [[price, qty], ...],
        "obi": 0.1,             # Order book imbalance (-1..1)
        "rv_short": 0.0005      # Short term realised volatility
    }

Only the top level of the book is used.  ``spread`` is derived from the
best bid/ask unless explicitly provided in the JSON.

Example
-------
::

    python -m backtest.tick_sim --data path/to.jsonl \
        --strategy maker_spread --maker_bp 16 --taker_bp 26 \
        --edge_margin_bp 3

The command prints a short performance summary and writes a ``trades.csv``
file containing individual trades and their PnL components.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

__all__ = ["Snapshot", "Trade", "run_sim", "main"]


@dataclass
class Snapshot:
    """Minimal representation of a market snapshot."""

    ts: float
    bid: float
    ask: float
    obi: float = 0.0
    rv_short: float = 0.0
    spread: float | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        if self.spread is None:
            self.spread = self.ask - self.bid


@dataclass
class Trade:
    """Result of a single backtest trade."""

    entry_ts: float
    exit_ts: float
    side: str
    entry_price: float
    exit_price: float
    qty: float
    fees: float
    pnl: float
    hold_secs: float


class TickSimulator:
    """Simulate a very small subset of maker/taker behaviour."""

    def __init__(
        self,
        snapshots: List[Snapshot],
        maker_bp: float,
        taker_bp: float,
        edge_margin_bp: float,
    ) -> None:
        self.snapshots = snapshots
        self.maker_bp = maker_bp / 10_000.0
        self.taker_bp = taker_bp / 10_000.0
        self.edge_margin_bp = edge_margin_bp / 10_000.0
        self.random = random.Random(0)

    # ------------------------------------------------------------------
    def run(self) -> List[Trade]:
        trades: List[Trade] = []
        position = 0  # -1 short, 1 long
        entry_snap: Snapshot | None = None
        pending_side: int | None = None
        pending_qty: float = 0.0
        pending_start: float = 0.0

        for snap in self.snapshots:
            # Handle pending maker order
            if pending_side is not None:
                wait = snap.ts - pending_start
                dir_obi = snap.obi if pending_side > 0 else -snap.obi
                fill_prob = min(1.0, 0.05 + 0.5 * max(dir_obi, 0) + 0.1 * wait)
                if self.random.random() < fill_prob:
                    # partial fill
                    fraction = self.random.uniform(0.5, 1.0)
                    pending_qty -= fraction
                    if pending_qty <= 0.0:
                        position = pending_side
                        entry_snap = snap
                        pending_side = None
                continue

            if position == 0:
                if snap.obi > self.edge_margin_bp:
                    pending_side = 1
                    pending_qty = 1.0
                    pending_start = snap.ts
                elif snap.obi < -self.edge_margin_bp:
                    pending_side = -1
                    pending_qty = 1.0
                    pending_start = snap.ts
            else:
                exit_now = (position == 1 and snap.obi < 0) or (
                    position == -1 and snap.obi > 0
                )
                if exit_now and entry_snap is not None:
                    slippage = snap.rv_short * snap.spread
                    if position == 1:
                        exit_price = snap.bid - slippage
                        entry_price = entry_snap.bid
                    else:
                        exit_price = snap.ask + slippage
                        entry_price = entry_snap.ask
                    maker_fee = entry_price * self.maker_bp
                    taker_fee = exit_price * self.taker_bp
                    fees = maker_fee + taker_fee
                    if position == 1:
                        pnl = exit_price - entry_price - fees
                    else:
                        pnl = entry_price - exit_price - fees
                    hold_secs = snap.ts - entry_snap.ts
                    trades.append(
                        Trade(
                            entry_snap.ts,
                            snap.ts,
                            "long" if position == 1 else "short",
                            entry_price,
                            exit_price,
                            1.0,
                            fees,
                            pnl,
                            hold_secs,
                        )
                    )
                    position = 0
                    entry_snap = None
        return trades


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_snapshots(path: Path) -> List[Snapshot]:
    """Return list of :class:`Snapshot` objects from ``path``."""

    items: List[Snapshot] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            bids = data.get("bids")
            asks = data.get("asks")
            if bids and asks:
                bid = float(bids[0][0])
                ask = float(asks[0][0])
            else:
                bid = float(data.get("bid", 0.0))
                ask = float(data.get("ask", 0.0))
            items.append(
                Snapshot(
                    ts=float(data.get("ts") or data.get("timestamp") or 0.0),
                    bid=bid,
                    ask=ask,
                    obi=float(data.get("obi", 0.0)),
                    rv_short=float(data.get("rv_short", 0.0)),
                    spread=float(data.get("spread")) if data.get("spread") else None,
                )
            )
    return items


def summarise(trades: List[Trade]) -> dict:
    """Return performance metrics for ``trades``."""

    if not trades:
        return {
            "expectancy": 0.0,
            "hit_rate": 0.0,
            "avg_hold_secs": 0.0,
            "avg_fee_drag": 0.0,
            "max_drawdown": 0.0,
        }

    pnls = [t.pnl for t in trades]
    expectancy = sum(pnls) / len(pnls)
    hit_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    avg_hold = sum(t.hold_secs for t in trades) / len(trades)
    avg_fee = sum(t.fees for t in trades) / len(trades)

    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "expectancy": expectancy,
        "hit_rate": hit_rate,
        "avg_hold_secs": avg_hold,
        "avg_fee_drag": avg_fee,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def run_sim(args: argparse.Namespace) -> pd.DataFrame:
    snaps = load_snapshots(Path(args.data))
    sim = TickSimulator(snaps, args.maker_bp, args.taker_bp, args.edge_margin_bp)
    trades = sim.run()
    df = pd.DataFrame(dataclasses.asdict(t) for t in trades)
    df.to_csv(args.out, index=False)
    metrics = summarise(trades)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tick level backtest simulator")
    p.add_argument("--data", required=True, help="Path to snapshot JSONL file")
    p.add_argument("--strategy", default="maker_spread", help="Strategy name")
    p.add_argument("--maker_bp", type=float, default=0.0, help="Maker fee in bps")
    p.add_argument("--taker_bp", type=float, default=0.0, help="Taker fee in bps")
    p.add_argument(
        "--edge_margin_bp",
        type=float,
        default=0.0,
        help="Edge margin threshold in bps",
    )
    p.add_argument("--out", default="trades.csv", help="Output CSV path")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    p = build_arg_parser()
    args = p.parse_args(list(argv) if argv is not None else None)
    run_sim(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio
import logging
import time
from typing import Dict, Any

import pandas as pd

from .utils import market_loader


async def control_loop(state: Dict[str, Any], ctx: Any | None = None, session_state: Any | None = None) -> None:
    """Listen for commands and update ``state`` accordingly."""
    print("Commands: start | stop | reload | panic sell | status | quit")
    try:
        while True:
            try:
                cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
            except EOFError:
                logging.warning("EOF on console input; exiting control loop")
                break
            if cmd == "start":
                state["running"] = True
                print("Trading started")
            elif cmd == "stop":
                state["running"] = False
                print("Trading stopped")
            elif cmd == "reload":
                state["reload"] = True
                print("Reloading config")
            elif cmd in {"panic", "panic sell", "panic_sell"}:
                state["liquidate_all"] = True
                print("Liquidation scheduled")
            elif cmd == "status":
                print_status_table(ctx, session_state)
            elif cmd in {"quit", "exit"}:
                state["running"] = False
                break
    except asyncio.CancelledError:
        state["running"] = False
        raise


def _fmt_last_update(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "-"
    try:
        idx = df.index[-1]
        if isinstance(idx, pd.Timestamp):
            return idx.isoformat()
        return pd.to_datetime(idx, unit="ms", utc=True).isoformat()
    except Exception:
        return "-"


def print_status_table(ctx: Any | None, session_state: Any | None) -> None:
    """Print a table of cache freshness and backoff state."""
    if ctx is None or session_state is None:
        print("No context available")
        return
    symbols = getattr(ctx, "active_universe", [])
    tfs = ctx.config.get("timeframes", [])
    headers = ["symbol", *tfs, "backoff"]
    rows: list[list[str]] = []
    now = time.time()
    for sym in symbols:
        row = [sym]
        for tf in tfs:
            df = session_state.df_cache.get(tf, {}).get(sym)
            row.append(_fmt_last_update(df))
        info = market_loader.failed_symbols.get(sym)
        if not info:
            row.append("-")
        elif info.get("disabled"):
            row.append("disabled")
        else:
            retry = int(info["time"] + info["delay"] - now)
            row.append(f"{retry}s" if retry > 0 else "-")
        rows.append(row)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = " ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    for r in rows:
        print(" ".join(r[i].ljust(widths[i]) for i in range(len(headers))))


from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Optional

import yaml


def load_execution_mode(config_file: Path) -> str:
    """Return execution mode from the YAML config."""
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f).get("execution_mode", "dry_run")
    return "dry_run"


def set_execution_mode(mode: str, config_file: Path) -> None:
    """Update execution mode in the YAML config."""
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    config["execution_mode"] = mode
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)


def compute_performance(df: pd.DataFrame) -> dict[str, float]:
    """Return realized PnL per symbol from the trades dataframe.

    Both long and short trades are supported. Sells first reduce any open
    long position and excess amount opens a short. Buys cover shorts before
    adding to the long side.
    """

    perf: dict[str, float] = {}
    open_longs: dict[str, list[tuple[float, float]]] = {}
    open_shorts: dict[str, list[tuple[float, float]]] = {}
    # Track open positions per symbol. Positive quantity represents a long
    # position while negative quantity represents a short position.  The FIFO
    # order of the list is preserved for proper PnL calculation.
    open_pos: dict[str, list[tuple[float, float]]] = {}

    for _, row in df.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        price = float(row.get("price", 0))
        amount = float(row.get("amount", 0))

        if side == "buy":
            # Close shorts first
            shorts = open_shorts.setdefault(symbol, [])
            while amount > 0 and shorts:
                entry_price, qty = shorts.pop(0)
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (entry_price - price) * traded
                if qty > traded:
                    shorts.insert(0, (entry_price, qty - traded))
                amount -= traded
            if amount > 0:
                open_longs.setdefault(symbol, []).append((price, amount))

        elif side == "sell":
            # Close longs first
            longs = open_longs.setdefault(symbol, [])
            while amount > 0 and longs:
                entry_price, qty = longs.pop(0)
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (price - entry_price) * traded
                if qty > traded:
                    longs.insert(0, (entry_price, qty - traded))
                amount -= traded
            if amount > 0:
                open_shorts.setdefault(symbol, []).append((price, amount))
        positions = open_pos.setdefault(symbol, [])

        if side == "buy":
            # Buys first close any existing short positions
            while amount > 0 and positions and positions[0][1] < 0:
                entry_price, qty = positions.pop(0)
                qty = -qty  # convert short quantity to positive
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (entry_price - price) * traded
                if qty > traded:
                    positions.insert(0, (entry_price, -(qty - traded)))
                amount -= traded
            # Remaining amount opens a new long position
            if amount > 0:
                positions.append((price, amount))

        elif side == "sell":
            # Sells first close existing long positions
            while amount > 0 and positions and positions[0][1] > 0:
                entry_price, qty = positions.pop(0)
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (price - entry_price) * traded
                if qty > traded:
                    positions.insert(0, (entry_price, qty - traded))
                amount -= traded
            # Excess amount starts a short position
            if amount > 0:
                positions.append((price, -amount))

    return perf


def is_running(proc: Optional[subprocess.Popen]) -> bool:
    """Return True if the given process is running."""
    return proc is not None and proc.poll() is None


def get_uptime(start_time: Optional[float]) -> str:
    """Return human-readable uptime from a start timestamp."""
    if start_time is None:
        return "-"
    delta = int(time.time() - start_time)
    hrs, rem = divmod(delta, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_last_trade(trade_file: Path) -> str:
    """Return last trade from trades CSV."""
    if not trade_file.exists():
        return "N/A"
    import csv

    with open(trade_file) as f:
        rows = list(csv.reader(f))
    if not rows:
        return "N/A"
    row = rows[-1]
    if len(row) >= 4:
        sym, side, amt, price = row[:4]
        return f"{side} {amt} {sym} @ {price}"
    return "N/A"


def get_current_regime(log_file: Path) -> str:
    """Return most recent regime classification from bot log."""
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        for line in reversed(lines):
            if "Market regime classified as" in line:
                return line.rsplit("Market regime classified as", 1)[1].strip()
    return "N/A"


def get_last_decision_reason(log_file: Path) -> str:
    """Return the last evaluation reason from bot log."""
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        for line in reversed(lines):
            if "[EVAL]" in line:
                return line.split("[EVAL]", 1)[1].strip()
    return "N/A"

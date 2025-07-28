from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict

from .logger import LOG_DIR


import pandas as pd

LOG_FILE = LOG_DIR / "pattern_frequency.csv"


def log_patterns(regime: str, patterns: Dict[str, float]) -> None:
    """Append detected ``patterns`` for ``regime`` to the CSV log."""
    if not patterns:
        return
    records = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "pattern": name,
            "strength": float(strength),
        }
        for name, strength in patterns.items()
    ]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)


def recent_pattern_strength(symbol: str, pattern: str) -> float:
    """Return the most recent strength for ``pattern`` on ``symbol``.

    Parameters
    ----------
    symbol:
        Token or trading pair to filter by.
    pattern:
        Pattern name like ``"volatile"``.

    Returns
    -------
    float
        Strength value from the latest matching row or ``0.0`` when not found.
    """

    if not LOG_FILE.exists():
        return 0.0

    if hasattr(pd, "read_csv"):
        try:
            df = pd.read_csv(LOG_FILE)
        except Exception:
            return 0.0

        if "token" in df.columns:
            df = df[df["token"] == symbol]
        elif "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        df = df[df["pattern"] == pattern]
        if df.empty:
            return 0.0

        df = df.sort_values("timestamp")
        return float(df.iloc[-1]["strength"])

    # Fallback when pandas is unavailable
    import csv

    try:
        with open(LOG_FILE, newline="") as f:
            rows = [row for row in csv.DictReader(f)]
    except Exception:
        return 0.0

    rows = [r for r in rows if (r.get("token") or r.get("symbol")) == symbol and r.get("pattern") == pattern]
    if not rows:
        return 0.0

    return float(rows[-1].get("strength", 0.0))


def average_pattern_strength(name: str, lookback: int = 1000) -> float:
    """Return the mean strength for ``name`` from the pattern log.

    Parameters
    ----------
    name:
        Pattern name to average, e.g. ``"volume_spike"``.
    lookback:
        Number of recent rows to include in the calculation. ``0`` means all
        rows.
    """

    if not LOG_FILE.exists():
        return 0.0

    if hasattr(pd, "read_csv"):
        try:
            df = pd.read_csv(LOG_FILE)
        except Exception:
            return 0.0

        df = df[df["pattern"] == name]
        if df.empty:
            return 0.0

        df = df.sort_values("timestamp")
        if lookback > 0:
            df = df.tail(int(lookback))
        return float(df["strength"].mean())

    import csv

    try:
        with open(LOG_FILE, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("pattern") == name]
    except Exception:
        return 0.0

    if lookback > 0:
        rows = rows[-int(lookback) :]
    if not rows:
        return 0.0

    vals = [float(r.get("strength", 0.0)) for r in rows]
    return sum(vals) / len(vals)

from __future__ import annotations

from fastapi import FastAPI
from pathlib import Path
import json
import re

app = FastAPI()

SIGNALS_FILE = Path("crypto_bot/logs/asset_scores.json")
POSITIONS_FILE = Path("crypto_bot/logs/positions.log")
PERFORMANCE_FILE = Path("crypto_bot/logs/strategy_performance.json")
SCORES_FILE = Path("crypto_bot/logs/strategy_scores.json")


@app.get("/live-signals")
def live_signals() -> dict:
    """Return latest signal scores as a mapping of symbol to score."""
    if SIGNALS_FILE.exists():
        try:
            return json.loads(SIGNALS_FILE.read_text())
        except Exception:
            return {}
    return {}


POS_PATTERN = re.compile(
    r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
    r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
    r"pnl \$(?P<pnl>[0-9.+-]+).*balance \$(?P<balance>[0-9.]+)"
)


@app.get("/positions")
def positions() -> list[dict]:
    """Return parsed position log entries."""
    if not POSITIONS_FILE.exists():
        return []
    entries: list[dict] = []
    for line in POSITIONS_FILE.read_text().splitlines():
        match = POS_PATTERN.search(line)
        if not match:
            continue
        entries.append(
            {
                "symbol": match.group("symbol"),
                "side": match.group("side"),
                "amount": float(match.group("amount")),
                "entry_price": float(match.group("entry")),
                "current_price": float(match.group("current")),
                "pnl": float(match.group("pnl")),
                "balance": float(match.group("balance")),
            }
        )
    return entries


@app.get("/strategy-performance")
def strategy_performance() -> dict:
    """Return raw strategy performance data grouped by regime and strategy."""
    if PERFORMANCE_FILE.exists():
        try:
            return json.loads(PERFORMANCE_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.get("/strategy-scores")
def strategy_scores() -> dict:
    """Return computed strategy metrics."""
    if SCORES_FILE.exists():
        try:
            return json.loads(SCORES_FILE.read_text())
        except Exception:
            return {}
    return {}

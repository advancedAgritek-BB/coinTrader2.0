"""Placeholder performance analytics helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd

from .logger import LOG_DIR


STATS_FILE = LOG_DIR / "strategy_stats.json"
PERFORMANCE_FILE = LOG_DIR / "strategy_performance.json"


def _load_stats() -> dict:
    """Return parsed stats file or empty dict on failure."""
    if not STATS_FILE.exists():
        return {}
    try:
        return json.loads(STATS_FILE.read_text())
    except Exception:
        return {}


def _extract_records(data: object, symbol: str | None) -> Sequence[dict] | None:
    """Return list of trade/return records from ``data``."""
    if isinstance(data, list):
        if symbol:
            return [r for r in data if isinstance(r, dict) and r.get("symbol") == symbol]
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # nested by symbol
        if symbol and symbol in data:
            return data[symbol] if isinstance(data[symbol], list) else None
        # maybe under 'history' or 'records'
        for key in (symbol, "history", "records"):
            if key and key in data and isinstance(data[key], list):
                return data[key]
    return None


def recompute() -> dict:
    """Recompute aggregate strategy metrics from performance log.

    Reads :data:`PERFORMANCE_FILE` containing raw trade history grouped by
    regime and strategy, calculates summary statistics per strategy and writes
    the result to :data:`STATS_FILE`.

    Returns the computed statistics mapping strategy name to metrics.
    """

    if not PERFORMANCE_FILE.exists():
        STATS_FILE.write_text("{}")
        return {}
    try:
        perf = json.loads(PERFORMANCE_FILE.read_text())
    except Exception:
        STATS_FILE.write_text("{}")
        return {}

    strat_trades: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for regime_data in perf.values():
        if not isinstance(regime_data, dict):
            continue
        for strat, trades in regime_data.items():
            if not isinstance(trades, list):
                continue
            bucket = strat_trades.setdefault(strat, [])
            for t in trades:
                if not isinstance(t, dict):
                    continue
                pnl = t.get("pnl")
                if pnl is None:
                    continue
                ts = t.get("timestamp") or t.get("exit_time") or t.get("time")
                try:
                    pnl_val = float(pnl)
                except Exception:
                    continue
                try:
                    ts_val = pd.to_datetime(ts, utc=True) if ts else pd.NaT
                except Exception:
                    ts_val = pd.NaT
                bucket.append((ts_val, pnl_val))

    stats: dict[str, dict[str, float | int]] = {}
    for strat, trades in strat_trades.items():
        if not trades:
            continue
        trades.sort(key=lambda x: x[0] if not pd.isna(x[0]) else pd.Timestamp.min)
        pnls = pd.Series([p for _, p in trades])
        count = int(len(pnls))
        wins = int((pnls > 0).sum())
        win_rate = wins / count if count else 0.0
        std = pnls.std(ddof=0)
        sharpe = float(pnls.mean() / std * (count ** 0.5)) if std else 0.0
        cum = pnls.cumsum()
        running_max = cum.cummax()
        drawdown = float((running_max - cum).max()) if count else 0.0
        avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
        avg_loss = float(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0.0
        stats[strat] = {
            "trades": count,
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "drawdown": float(drawdown),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "total_pnl": float(pnls.sum()),
        }

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def edge(strategy: str, symbol: str, coef: float = 0.3) -> float:
    """Return the performance edge for ``strategy`` and ``symbol``.

    The function looks up historical stats stored in
    ``crypto_bot/logs/strategy_stats.json``. It calculates the Sharpe ratio and
    maximum drawdown of the PnL records over the last 30 days and returns
    ``sharpe - coef * drawdown``. If the file is missing or malformed ``0.0`` is
    returned.
    """

    data = _load_stats()
    strat_data = data.get(strategy)
    if not strat_data:
        return 0.0

    records = _extract_records(strat_data, symbol)
    if not records:
        return 0.0

    now = datetime.utcnow()
    start = now - timedelta(days=30)
    values: list[float] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        pnl = rec.get("pnl")
        if pnl is None:
            pnl = rec.get("return") or rec.get("pnl_pct")
        if pnl is None:
            continue
        ts = (
            rec.get("timestamp")
            or rec.get("time")
            or rec.get("date")
            or rec.get("exit_time")
        )
        if ts:
            try:
                t = pd.to_datetime(ts, utc=True)
            except Exception:
                t = None
        else:
            t = None
        if t is not None and t < start:
            continue
        try:
            values.append(float(pnl))
        except Exception:
            continue

    if not values:
        # fallback to last 30 records if timestamps missing
        all_vals = [float(rec.get("pnl", 0.0)) for rec in records if isinstance(rec, dict) and "pnl" in rec]
        values = all_vals[-30:]
    if not values:
        return 0.0

    series = pd.Series(values)
    std = series.std()
    sharpe = float(series.mean() / std * (len(series) ** 0.5)) if std else 0.0

    cum = series.cumsum()
    running_max = cum.cummax()
    drawdown = float((running_max - cum).max())

    return sharpe - coef * drawdown

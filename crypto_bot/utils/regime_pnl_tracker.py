import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .logger import LOG_DIR

LOG_FILE = LOG_DIR / "regime_pnl.csv"
# JSON log used by meta_selector for historical strategy performance
PERF_FILE = LOG_DIR / "strategy_performance.json"


def _seed_fake_trades(path: str | Path = LOG_FILE) -> None:
    """Populate ``path`` with fake trades if the file is empty."""
    file = Path(path)
    if file.exists():
        try:
            df = pd.read_csv(file)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            return
    rows = []
    for _ in range(20):
        rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "regime": "breakout",
                "strategy": "sniper_solana",
                "pnl": 0.02,
            }
        )
    for _ in range(50):
        rows.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "regime": "volatile",
                "strategy": "sniper_solana",
                "pnl": 0.05,
            }
        )
    file.parent.mkdir(parents=True, exist_ok=True)
    header = not file.exists() or file.stat().st_size == 0
    pd.DataFrame(rows).to_csv(file, mode="a", header=header, index=False)


def log_trade(regime: str, strategy: str, pnl: float) -> None:
    """Append realized PnL for ``strategy`` in ``regime`` to the logs."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "regime": regime,
        "strategy": strategy,
        "pnl": float(pnl),
    }
    # Write to the CSV file used for quick metric calculations
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)

    # Maintain JSON log grouped by regime and strategy for meta_selector
    PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(PERF_FILE.read_text()) if PERF_FILE.exists() else {}
    except Exception:
        data = {}
    entry = {"timestamp": record["timestamp"], "pnl": float(pnl)}
    regime_block = data.setdefault(regime, {})
    regime_block.setdefault(strategy, []).append(entry)
    PERF_FILE.write_text(json.dumps(data))


def _calc_metrics(pnls: pd.Series) -> Dict[str, float]:
    equity = pnls.cumsum()
    running_max = equity.cummax()
    drawdown = (1 - equity / running_max).max() if not equity.empty else 0.0
    sharpe = 0.0
    if len(pnls) > 1 and pnls.std() != 0:
        sharpe = pnls.mean() / pnls.std() * (len(pnls) ** 0.5)
    return {
        "pnl": float(pnls.sum()),
        "sharpe": float(sharpe),
        "drawdown": float(drawdown),
    }


def get_metrics(regime: str | None = None, path: str | Path = LOG_FILE) -> Dict[str, Dict[str, Any]]:
    """Return PnL metrics grouped by regime and strategy."""
    file = Path(path)
    if not file.exists():
        return {}
    df = pd.read_csv(file)
    if df.empty:
        return {}
    if regime:
        df = df[df["regime"] == regime]
    metrics: Dict[str, Dict[str, Any]] = {}
    for (reg, strat), group in df.groupby(["regime", "strategy"]):
        stats = _calc_metrics(group["pnl"])
        metrics.setdefault(reg, {})[strat] = stats
    return metrics


def compute_weights(regime: str, path: str | Path = LOG_FILE) -> Dict[str, float]:
    """Return normalized strategy weights for ``regime`` using Sharpe ratio."""
    data = get_metrics(regime, path)
    strategies = data.get(regime, {})
    if not strategies:
        return {}
    scores = {s: m["sharpe"] for s, m in strategies.items()}
    total = sum(scores.values())
    if not total:
        return {s: 1 / len(scores) for s in scores}
    return {s: sc / total for s, sc in scores.items()}


def get_recent_win_rate(
    window: int = 20,
    path: str | Path = LOG_FILE,
    strategy: str | None = None,
    half_life: float | None = 5.0,
) -> float:
    """Return the fraction of profitable trades with optional decay weighting.

    Parameters
    ----------
    window : int, optional
        Number of most recent trades to evaluate (default ``20``).
    path : str or Path, optional
        CSV log file location (defaults to :data:`LOG_FILE`).
    strategy : str, optional
        If given, filter trades for the specified strategy.
    half_life : float, optional
        Number of trades after which the contribution of a trade is halved.
        ``None`` disables weighting.

    Returns
    -------
    float
        Win rate over the evaluated trades. Newer trades count more when
        ``half_life`` is provided.
    """
    file = Path(path)
    if not file.exists():
        return 0.6
    df = pd.read_csv(file)
    if df.empty:
        return 0.6
    if strategy is not None and "strategy" in df.columns:
        df = df[df["strategy"] == strategy]

    recent = df.tail(window)
    if strategy is not None and "strategy" in recent.columns:
        recent = recent[recent["strategy"] == strategy]

    outcomes = (recent["pnl"] > 0).astype(float)
    total = len(outcomes)
    if not total:
        return 0.0

    if half_life and half_life > 0:
        ages = np.arange(total - 1, -1, -1)
        weights = 0.5 ** (ages / half_life)
        weights /= weights.sum()
        return float((outcomes * weights).sum())

    wins = outcomes.sum()
    return float(wins / total)


# Seed the PnL log with fake trades when empty to bootstrap weights
# _seed_fake_trades(LOG_FILE)

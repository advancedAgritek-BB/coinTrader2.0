import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .logger import LOG_DIR


LOG_FILE = LOG_DIR / "regime_pnl.csv"


def log_trade(regime: str, strategy: str, pnl: float) -> None:
    """Append realized PnL for ``strategy`` in ``regime`` to the CSV log."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "regime": regime,
        "strategy": strategy,
        "pnl": float(pnl),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)


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
) -> float:
    """Return win rate over the last ``window`` trades.

    Parameters
    ----------
    window : int, optional
        Number of most recent trades to evaluate. Defaults to ``20``.
    path : str | Path, optional
        CSV log file path. Defaults to :data:`LOG_FILE`.
    strategy : str, optional
        Filter trades by strategy name if provided.
    strategy : str | None, optional
        Filter trades to this strategy name when provided.
    """
    file = Path(path)
    if not file.exists():
        return 0.0
    df = pd.read_csv(file)
    if df.empty:
        return 0.0
    if strategy and "strategy" in df.columns:
    if strategy is not None:
        df = df[df["strategy"] == strategy]
    recent = df.tail(window)
    wins = (recent["pnl"] > 0).sum()
    total = len(recent)
    return float(wins / total) if total else 0.0

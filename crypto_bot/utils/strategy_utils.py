import pandas as pd
from pathlib import Path
from typing import Dict

from .logger import LOG_DIR



def compute_strategy_weights(path: str = str(LOG_DIR / "strategy_pnl.csv")) -> Dict[str, float]:
    """Return normalized allocation weights per strategy.

    We compute either win rate or Sharpe ratio from the PnL data stored at
    ``path``. The file must contain ``strategy`` and ``pnl`` columns. If the
    file is missing or empty an empty dict is returned.
    """
    file = Path(path)
    if not file.exists():
        return {}

    df = pd.read_csv(file)
    if df.empty or "strategy" not in df.columns or "pnl" not in df.columns:
        return {}

    scores = {}
    for strat, group in df.groupby("strategy"):
        wins = (group["pnl"] > 0).sum()
        total = len(group)
        win_rate = wins / total if total else 0.0
        std = group["pnl"].std()
        sharpe = group["pnl"].mean() / std * (total ** 0.5) if std else 0.0
        scores[strat] = max(win_rate, sharpe)

    total_score = sum(scores.values())
    if not total_score:
        return {s: 1 / len(scores) for s in scores} if scores else {}

    return {s: sc / total_score for s, sc in scores.items()}


def compute_drawdown(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return maximum drawdown of ``close`` prices over ``lookback`` bars."""
    if df.empty or "close" not in df.columns:
        return 0.0
    series = df["close"].tail(lookback)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdowns = series - running_max
    return float(drawdowns.min())

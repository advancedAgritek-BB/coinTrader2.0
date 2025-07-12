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


def compute_edge(
    strategy: str,
    drawdown_penalty: float = 0.0,
    path: str = str(LOG_DIR / "strategy_pnl.csv"),
) -> float:
    """Return the edge score for ``strategy`` based on past trades.

    The CSV located at ``path`` must contain ``strategy`` and ``pnl`` columns.
    The edge is calculated as::

        win_rate * (avg_gain / avg_loss) - drawdown_penalty * abs(drawdown)

    where ``drawdown`` is measured on the cumulative PnL series using
    :func:`compute_drawdown`.
    """

    file = Path(path)
    if not file.exists():
        return 0.0

    df = pd.read_csv(file)
    if df.empty or {"strategy", "pnl"}.difference(df.columns):
        return 0.0

    strat_df = df[df["strategy"] == strategy]
    if strat_df.empty:
        return 0.0

    pnl_series = strat_df["pnl"].astype(float)
    total = len(pnl_series)
    wins = (pnl_series > 0).sum()
    losses = (pnl_series < 0).sum()

    win_rate = wins / total if total else 0.0
    avg_gain = pnl_series[pnl_series > 0].mean() if wins else 0.0
    avg_loss = (-pnl_series[pnl_series < 0]).mean() if losses else 0.0

    ratio = (avg_gain / avg_loss) if avg_loss else 0.0

    cumulative = pnl_series.cumsum()
    drawdown = compute_drawdown(pd.DataFrame({"close": cumulative}))

    return win_rate * ratio - drawdown_penalty * abs(drawdown)


__all__ = ["compute_strategy_weights", "compute_drawdown", "compute_edge"]

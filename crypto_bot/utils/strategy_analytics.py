import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Default location for recorded trade performance. Each trade closed
# is appended to this JSON file via ``log_performance``.
STATS_FILE = Path("crypto_bot/logs/strategy_performance.json")
SCORES_FILE = Path("crypto_bot/logs/strategy_scores.json")


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def compute_metrics(path: Path = STATS_FILE) -> Dict[str, Dict[str, float]]:
    """Return Sharpe ratio, win rate, drawdown and EV for each strategy.

    The statistics file must contain a mapping of strategy name to a list of
    trade records with a ``pnl`` field::

        {
            "trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}],
            "grid_bot": [{"pnl": 0.2}]
        }
    """

    data = _load(path)
    metrics: Dict[str, Dict[str, float]] = {}
    for strat, trades in data.items():
        if not isinstance(trades, list):
            raise ValueError(
                f"Expected list of trade records for strategy '{strat}', got {type(trades).__name__}"
            )
        pnls = []
        for t in trades:
            if not isinstance(t, dict) or "pnl" not in t:
                raise ValueError(
                    "Each trade must be a mapping with a 'pnl' key. "
                    f"Got {t!r} for strategy '{strat}'."
                )
            pnls.append(float(t["pnl"]))
        if not pnls:
            metrics[strat] = {"sharpe": 0.0, "win_rate": 0.0, "drawdown": 0.0, "ev": 0.0}
            continue
        series = pd.Series(pnls)
        mean = series.mean()
        std = series.std()
        sharpe = float(mean / std * (len(series) ** 0.5)) if std else 0.0
        win_rate = float(sum(p > 0 for p in pnls) / len(pnls))
        cum = series.cumsum()
        running_max = cum.cummax()
        drawdown = float((cum - running_max).min())
        metrics[strat] = {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "drawdown": drawdown,
            "ev": float(mean),
        }
    return metrics


def write_scores(
    out_path: Path = SCORES_FILE, stats_path: Path = STATS_FILE
) -> Dict[str, Dict[str, float]]:
    """Compute metrics from ``stats_path`` and write them to ``out_path``."""

    scores = compute_metrics(stats_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores))
    return scores

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

    ``strategy_performance.json`` may store trade records in two formats:

    1. A direct mapping of strategy name to a list of trades::

        {
            "trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}]
        }

    2. Nested by market regime and strategy::

        {
            "trending": {
                "trend_bot": [{"pnl": 1.0}]
            }
        }

    This function normalizes either layout into per-strategy metrics.
    """

    data = _load(path)
    metrics: Dict[str, Dict[str, float]] = {}

    def _process(strategy: str, trades: Any) -> None:
        if not isinstance(trades, list):
            raise ValueError(
                f"Expected list of trade records for strategy '{strategy}', got {type(trades).__name__}"
            )

        pnls = []
        for rec in trades:
            if not isinstance(rec, dict) or "pnl" not in rec:
                raise ValueError(
                    "Each trade must be a mapping with a 'pnl' key. "
                    f"Got {rec!r} for strategy '{strategy}'."
                )
            pnls.append(float(rec["pnl"]))

        if not pnls:
            metrics[strategy] = {"sharpe": 0.0, "win_rate": 0.0, "drawdown": 0.0, "ev": 0.0}
            return

        series = pd.Series(pnls)
        mean = series.mean()
        std = series.std()
        sharpe = float(mean / std * (len(series) ** 0.5)) if std else 0.0
        win_rate = float(sum(p > 0 for p in pnls) / len(pnls))
        cum = series.cumsum()
        running_max = cum.cummax()
        drawdown = float((cum - running_max).min())
        metrics[strategy] = {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "drawdown": drawdown,
            "ev": float(mean),
        }

    for key, value in data.items():
        if isinstance(value, list):
            _process(key, value)
        elif isinstance(value, dict):
            for strat, trades in value.items():
                _process(strat, trades)
        else:
            raise ValueError(
                f"Expected list or dict for entry '{key}', got {type(value).__name__}"
            )

    return metrics


def write_scores(
    out_path: Path = SCORES_FILE, stats_path: Path = STATS_FILE
) -> Dict[str, Dict[str, float]]:
    """Compute metrics from ``stats_path`` and write them to ``out_path``."""

    scores = compute_metrics(stats_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores))
    return scores

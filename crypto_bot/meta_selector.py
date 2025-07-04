import json
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    dex_scalper,
    mean_bot,
    breakout_bot,
    micro_scalp_bot,
)

LOG_FILE = Path("crypto_bot/logs/strategy_performance.json")


_STRATEGY_FN_MAP = {
    "trend": trend_bot.generate_signal,
    "trend_bot": trend_bot.generate_signal,
    "grid": grid_bot.generate_signal,
    "grid_bot": grid_bot.generate_signal,
    "sniper": sniper_bot.generate_signal,
    "sniper_bot": sniper_bot.generate_signal,
    "dex_scalper": dex_scalper.generate_signal,
    "dex_scalper_bot": dex_scalper.generate_signal,
    "mean_bot": mean_bot.generate_signal,
    "breakout_bot": breakout_bot.generate_signal,
    "micro_scalp": micro_scalp_bot.generate_signal,
    "micro_scalp_bot": micro_scalp_bot.generate_signal,
}


def _load() -> Dict[str, Dict[str, List[dict]]]:
    """Return parsed performance log data."""
    if not LOG_FILE.exists():
        return {}
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return {}


def _scores_for(regime: str) -> Dict[str, float]:
    """Compute score per strategy for ``regime``."""
    data = _load().get(regime, {})
    scores: Dict[str, float] = {}
    for strat, trades in data.items():
        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        if not pnls:
            continue
        wins = sum(p > 0 for p in pnls)
        total = len(pnls)
        win_rate = wins / total if total else 0.0
        series = pd.Series(pnls)
        sharpe = 0.0
        std = series.std()
        if std:
            sharpe = series.mean() / std * (total ** 0.5)
        scores[strat] = max(win_rate, sharpe)
    return scores


def choose_best(regime: str) -> Callable[[pd.DataFrame], tuple[float, str]]:
    """Return strategy with best historical score for ``regime``."""
    from .strategy_router import strategy_for

    scores = _scores_for(regime)
    if not scores:
        return strategy_for(regime)
    best = max(scores.items(), key=lambda x: x[1])[0]
    return _STRATEGY_FN_MAP.get(best, strategy_for(regime))

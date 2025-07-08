import json
from pathlib import Path
from typing import Callable, Dict, List
from datetime import datetime

import pandas as pd

from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    dex_scalper,
    mean_bot,
    breakout_bot,
    micro_scalp_bot,
    bounce_scalper,
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
    "bounce_scalper": bounce_scalper.generate_signal,
    "bounce_scalper_bot": bounce_scalper.generate_signal,
}


def get_strategy_by_name(
    name: str,
) -> Callable[[pd.DataFrame], tuple] | None:
    """Return the strategy function mapped to ``name`` if present."""
    return _STRATEGY_FN_MAP.get(name)


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
        now = datetime.utcnow()
        pnls = [
            float(t["pnl"]) * (0.98 ** (now - datetime.fromisoformat(t["timestamp"])).days)
            for t in trades
        ]
        if not pnls:
            continue
        wins = sum(p > 0 for p in pnls)
        total = len(pnls)
        win_rate = wins / total if total else 0.0
        series = pd.Series(pnls)

        neg_returns = series[series < 0]
        downside_std = neg_returns.std(ddof=0) if not neg_returns.empty else 0.0
        max_dd = (series.cummax() - series).max()

        raw_sharpe = 0.0
        std = series.std()
        if std:
            raw_sharpe = series.mean() / std * (total ** 0.5)

        score = win_rate * raw_sharpe / (1 + downside_std + max_dd)
        scores[strat] = score
    return scores


def choose_best(regime: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy with best historical score for ``regime``."""
    from .strategy_router import strategy_for

    scores = _scores_for(regime)
    if not scores:
        return strategy_for(regime)
    best = max(scores.items(), key=lambda x: x[1])[0]
    return _STRATEGY_FN_MAP.get(best, strategy_for(regime))

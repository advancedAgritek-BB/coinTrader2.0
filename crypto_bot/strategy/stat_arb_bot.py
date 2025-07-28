"""Simple statistical arbitrage pair trading strategy."""

from typing import Optional, Tuple

import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("stat_arb_bot")
except Exception:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return (score, direction) for pair trading opportunities."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("stat_arb", {}) if config else {}
    lookback = int(params.get("lookback", 20))
    threshold = float(params.get("zscore_threshold", 2.0))
    price_col = params.get("price_col", "close")
    benchmark_col = params.get("benchmark_col", "benchmark_close")

    if benchmark_col not in df.columns or price_col not in df.columns:
        return 0.0, "none"

    ratio = df[price_col] / df[benchmark_col]
    mean = ratio.rolling(lookback).mean().shift(1)
    std = ratio.rolling(lookback).std().shift(1)

    mean = cache_series("stat_arb_mean", df, mean, lookback)
    std = cache_series("stat_arb_std", df, std, lookback)

    recent = df.iloc[-(lookback + 1) :].copy()
    recent["mean"] = mean.iloc[-(lookback + 1) :]
    recent["std"] = std.iloc[-(lookback + 1) :]

    if recent["std"].iloc[-1] == 0 or pd.isna(recent["std"].iloc[-1]):
        return 0.0, "none"

    z = (ratio.iloc[-1] - recent["mean"].iloc[-1]) / recent["std"].iloc[-1]

    score = 0.0
    direction = "none"
    if z > threshold:
        score = min(z / threshold, 1.0)
        direction = "short"
    elif z < -threshold:
        score = min(-z / threshold, 1.0)
        direction = "long"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"

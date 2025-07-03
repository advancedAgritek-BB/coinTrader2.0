import pandas as pd
from typing import Tuple, Optional
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Short-term momentum strategy using EMA divergence on DEX pairs."""
    if df.empty:
        return 0.0, "none"

    params = config.get("dex_scalper", {}) if config else {}
    fast_window = params.get("ema_fast", 5)
    slow_window = params.get("ema_slow", 20)
    min_score = params.get("min_signal_score", 0.1)

    if len(df) < slow_window:
        return 0.0, "none"

    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=fast_window)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=slow_window)

    latest = df.iloc[-1]
    if (
        latest["close"] == 0
        or pd.isnull(latest["ema_fast"])
        or pd.isnull(latest["ema_slow"])
    ):
        return 0.0, "none"

    momentum = latest["ema_fast"] - latest["ema_slow"]
    score = min(abs(momentum) / latest["close"], 1.0)

    if score < min_score:
        return 0.0, "none"

    if momentum > 0:
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    elif momentum < 0:
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"
    return 0.0, "none"

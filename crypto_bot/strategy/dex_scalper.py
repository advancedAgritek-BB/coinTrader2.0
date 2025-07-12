import pandas as pd
from typing import Tuple, Optional
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.volatility_filter import calc_atr

DEFAULT_PAIRS = ["BTC/USD", "ETH/USD"]
ALLOWED_PAIRS = load_liquid_pairs() or DEFAULT_PAIRS


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Short-term momentum strategy using EMA divergence on DEX pairs."""
    if df.empty:
        return 0.0, "none"

    symbol = config.get("symbol") if config else ""
    if symbol and ALLOWED_PAIRS and symbol not in ALLOWED_PAIRS:
        return 0.0, "none"

    params = config.get("dex_scalper", {}) if config else {}
    fast_window = params.get("ema_fast", 5)
    slow_window = params.get("ema_slow", 20)
    min_score = params.get("min_signal_score", 0.1)
    min_atr_pct = float(params.get("min_atr_pct", 0.0))

    if len(df) < max(fast_window, slow_window):
        return 0.0, "none"

    lookback = slow_window
    recent = df.iloc[-(lookback + 1) :]

    ema_fast = ta.trend.ema_indicator(recent["close"], window=fast_window)
    ema_slow = ta.trend.ema_indicator(recent["close"], window=slow_window)

    ema_fast = cache_series("ema_fast", df, ema_fast, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow, lookback)

    df = recent.copy()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow

    latest = df.iloc[-1]
    if (
        latest["close"] == 0
        or pd.isnull(latest["ema_fast"])
        or pd.isnull(latest["ema_slow"])
    ):
        return 0.0, "none"

    momentum = latest["ema_fast"] - latest["ema_slow"]
    score = min(abs(momentum) / latest["close"], 1.0)

    if min_atr_pct:
        if {"high", "low", "close"}.issubset(df.columns):
            atr = calc_atr(df)
        else:
            atr = 0.0
        if latest["close"] == 0 or pd.isna(atr) or atr / latest["close"] < min_atr_pct:
            return 0.0, "none"

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

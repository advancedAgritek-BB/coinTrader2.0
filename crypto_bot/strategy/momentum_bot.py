from typing import Optional, Tuple

import pandas as pd
import ta


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Simple momentum strategy using EMA divergence."""
    if df is None or df.empty:
        return 0.0, "none"

    cfg = config or {}
    fast = int(cfg.get("momentum_ema_fast", 5))
    slow = int(cfg.get("momentum_ema_slow", 20))

    ema_fast = ta.trend.ema_indicator(df["close"], window=fast)
    ema_slow = ta.trend.ema_indicator(df["close"], window=slow)

    latest_fast = ema_fast.iloc[-1]
    latest_slow = ema_slow.iloc[-1]
    latest_price = df["close"].iloc[-1]

    score = 0.0
    if latest_price:
        score = min(abs(latest_fast - latest_slow) / latest_price, 1.0)

    if latest_fast > latest_slow:
        return score, "long"
    if latest_fast < latest_slow:
        return score, "short"
    return 0.0, "none"

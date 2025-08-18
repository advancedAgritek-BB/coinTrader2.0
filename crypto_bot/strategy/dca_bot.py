from typing import Optional, Tuple

import pandas as pd

NAME = "dca_bot"

def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_,
) -> Tuple[float, str]:
    """Simple dollar-cost averaging signal supporting long and short."""
    if df is None or df.empty or "close" not in df:
        return 0.0, "none"

    ma = df["close"].rolling(20).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if price < ma * 0.9:
        return 0.8, "long"
    if price > ma * 1.1:
        return 0.8, "short"
    return 0.0, "none"


class regime_filter:
    """DCA bot works across regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return True

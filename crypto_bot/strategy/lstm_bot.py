from typing import Optional, Tuple

import pandas as pd


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Simple placeholder LSTM-based strategy signal."""
    if df is None or df.empty or "close" not in df:
        return 0.0, "none"
    ma = df["close"].rolling(10).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if price > ma:
        return 0.5, "long"
    if price < ma:
        return 0.5, "short"
    return 0.0, "none"


class regime_filter:
    """Run across all regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return True

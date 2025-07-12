"""Simple dollar-cost averaging strategy."""
from typing import Optional, Tuple

import pandas as pd


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return long signal when price is 5% below 20-period moving average."""
    if df is None or df.empty:
        return 0.0, "none"

    ma = df["close"].rolling(20).mean().iloc[-1]
    if df["close"].iloc[-1] < ma * 0.95:
        return 0.8, "long"
    return 0.0, "none"

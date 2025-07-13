from typing import Optional, Tuple

import pandas as pd


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Simple dollar-cost averaging signal."""
    if df is None or df.empty:
        return 0.0, "none"

    ma = df["close"].rolling(20).mean().iloc[-1]
    if df["close"].iloc[-1] < ma * 0.95:
        return 0.8, "long"
    return 0.0, "none"

import pandas as pd
from typing import Optional, Tuple


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return signal when a potential flash crash is detected."""
    if df is None or df.empty or "close" not in df:
        return 0.0, "none"

    if len(df) >= 2:
        prev = df["close"].iloc[-2]
        last = df["close"].iloc[-1]
        if last < prev * 0.9:
            return 1.0, "long"
    return 0.0, "none"

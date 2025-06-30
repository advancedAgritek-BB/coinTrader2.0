import pandas as pd
from typing import Tuple


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """Placeholder DEX scalping strategy."""
    if df['close'].pct_change().iloc[-1] > 0.01:
        return 0.6, 'long'
    elif df['close'].pct_change().iloc[-1] < -0.01:
        return 0.6, 'short'
    return 0.0, 'none'

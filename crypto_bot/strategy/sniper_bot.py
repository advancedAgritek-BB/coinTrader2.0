import pandas as pd
from typing import Tuple


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """Placeholder sniping logic for new token launches."""
    latest = df.iloc[-1]
    if latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 3:
        return 1.0, 'long'
    return 0.0, 'none'

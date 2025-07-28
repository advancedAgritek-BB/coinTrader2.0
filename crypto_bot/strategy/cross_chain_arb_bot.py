"""Cross-chain arbitrage strategy placeholder."""

from typing import Optional, Tuple
import pandas as pd


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return (score, direction) for potential arbitrage opportunity.

    Currently a minimal implementation that always returns no signal. Integrate
    with exchange price feeds to produce real arbitrage signals.
    """
    return 0.0, "none"

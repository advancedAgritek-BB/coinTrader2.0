from __future__ import annotations

from datetime import datetime

import pandas as pd

from .logger import LOG_DIR

LOG_FILE = LOG_DIR / "second_place.csv"


def log_second_place(symbol: str, regime: str, strat_name: str, score: float, edge_val: float) -> None:
    """Append ranking info for second place strategy to ``LOG_FILE``."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "regime": regime,
        "strategy": strat_name,
        "score": float(score),
        "edge": float(edge_val),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)

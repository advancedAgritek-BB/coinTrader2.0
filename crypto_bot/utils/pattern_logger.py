from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict

from .logger import LOG_DIR


import pandas as pd

LOG_FILE = LOG_DIR / "pattern_frequency.csv"


def log_patterns(regime: str, patterns: Dict[str, float]) -> None:
    """Append detected ``patterns`` for ``regime`` to the CSV log."""
    if not patterns:
        return
    records = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "pattern": name,
            "strength": float(strength),
        }
        for name, strength in patterns.items()
    ]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)

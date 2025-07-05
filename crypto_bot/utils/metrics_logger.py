import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

LOG_FILE = Path("crypto_bot/logs/metrics.csv")


def log_cycle(symbol_time: float, ohlcv_time: float, analyze_time: float,
              total_time: float, path: Optional[Path] = None) -> None:
    """Append cycle timing information to a CSV log."""
    file = Path(path or LOG_FILE)
    file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        [datetime.utcnow().isoformat(), symbol_time, ohlcv_time, analyze_time, total_time]
    ], columns=["timestamp", "symbol_time", "ohlcv_time", "analyze_time", "total_time"])
    header = not file.exists()
    df.to_csv(file, mode="a", header=header, index=False)
from typing import Dict, Any


def log_metrics_to_csv(metrics: Dict[str, Any], filename: str) -> None:
    """Append metrics dict to ``filename`` as a CSV row.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Mapping of metric names to values.
    filename : str
        Path to the CSV file to append to.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

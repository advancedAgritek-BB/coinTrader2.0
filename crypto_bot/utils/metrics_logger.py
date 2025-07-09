import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from .logger import LOG_DIR

LOG_FILE = LOG_DIR / "metrics.csv"


def log_cycle(
    symbol_time: float,
    ohlcv_time: float,
    analyze_time: float,
    total_time: float,
    ohlcv_fetch_latency: float = 0.0,
    execution_latency: float = 0.0,
    path: Optional[Path] = None,
) -> None:
    """Append cycle timing information to a CSV log."""
    file = Path(path or LOG_FILE)
    file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol_time": symbol_time,
                "ohlcv_time": ohlcv_time,
                "analyze_time": analyze_time,
                "total_time": total_time,
                "ohlcv_fetch_latency": ohlcv_fetch_latency,
                "execution_latency": execution_latency,
            }
        ]
    )
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

import pandas as pd
from pathlib import Path
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

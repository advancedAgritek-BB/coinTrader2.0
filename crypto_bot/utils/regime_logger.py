import pandas as pd
from pathlib import Path
from typing import Dict

from .logger import LOG_DIR


"""Simple regime logging utilities."""

LOG_FILE = LOG_DIR / "regime_log.csv"


def log_regime(symbol: str, regime: str, future_return: float) -> None:
    """Append regime observation to ``LOG_FILE``.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    regime : str
        Regime label returned by the classifier.
    future_return : float
        Percent change over the evaluation window.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [[symbol, regime, future_return]],
        columns=["symbol", "regime", "future_return"],
    )
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)


def summarize_accuracy(path: str | Path = LOG_FILE) -> Dict[str, float]:
    """Return mean future return grouped by regime."""
    file = Path(path)
    if not file.exists():
        return {}
    df = pd.read_csv(file, header=None, names=["symbol", "regime", "future_return"])
    grouped = df.groupby("regime")["future_return"].mean()
    return grouped.to_dict()

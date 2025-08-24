from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .telemetry import telemetry

logger = logging.getLogger(__name__)

# Track symbols already logged for missing OHLCV to avoid spamming
_LOGGED: set[str] = set()


def ensure_ohlcv(symbol: str, df: Optional[pd.DataFrame], metric: str = "signals.missing_ohlcv") -> bool:
    """Return ``True`` if ``df`` contains OHLCV data.

    Parameters
    ----------
    symbol:
        Trading pair identifier used for logging.
    df:
        OHLCV data frame which may be ``None`` or empty.
    metric:
        Telemetry counter name incremented when data is missing.

    The function logs a warning only once per ``symbol`` to prevent log spam
    and increments the provided telemetry ``metric`` each time it is invoked
    with missing data.
    """
    if df is None or df.empty:
        telemetry.inc(metric)
        sym = symbol or "unknown"
        if sym not in _LOGGED:
            logger.warning("No OHLCV data for %s; skipping", sym)
            _LOGGED.add(sym)
        return False
    return True

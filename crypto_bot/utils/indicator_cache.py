from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

# Global cache storing recent indicator values
CACHE: Dict[str, pd.Series] = {}


def _cache_key(df: pd.DataFrame, name: str, symbol: Optional[str] = None) -> str:
    """Return cache key for indicator ``name`` on ``df`` or ``symbol``."""
    ident = symbol if symbol is not None else str(id(df))
    return f"{ident}:{name}"


def cache_series(
    name: str,
    df: pd.DataFrame,
    series: pd.Series,
    lookback: int,
    symbol: Optional[str] = None,
) -> pd.Series:
    """Store ``series`` under ``name`` and return the cached version."""
    key = _cache_key(df, name, symbol)
    prev = CACHE.get(key)
    if prev is not None:
        combined = pd.concat([prev, series.iloc[-1:]])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = series
    combined = combined.tail(lookback + 1)
    CACHE[key] = combined
    return combined

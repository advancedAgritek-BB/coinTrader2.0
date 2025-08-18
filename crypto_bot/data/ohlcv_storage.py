"""Utility functions for persisting OHLCV data to Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _build_path(storage_path: str | Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    """Return the Parquet file path for the given parameters."""
    safe_symbol = symbol.replace("/", "-")
    return Path(storage_path) / exchange_id / f"{safe_symbol}_{timeframe}.parquet"


def _to_seconds(series: pd.Series) -> pd.Series:
    """Return ``series`` converted to integer seconds."""
    if pd.api.types.is_datetime64_any_dtype(series):
        secs = series.view("int64") // 10**9
    else:
        secs = pd.to_numeric(series, errors="coerce").astype("int64")
        if (secs > 1_000_000_000_000).any():  # milliseconds
            secs = secs // 1000
    return secs


def load_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    storage_path: str | Path,
) -> pd.DataFrame | None:
    """Load OHLCV data from disk.

    Returns a DataFrame sorted by ``timestamp`` in ascending order with the
    ``timestamp`` column expressed in seconds. If the file does not exist, ``None``
    is returned.
    """

    path = _build_path(storage_path, exchange_id, symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to read %s", path, exc_info=True)
        return None
    df = df.copy()
    df["timestamp"] = _to_seconds(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    storage_path: str | Path,
) -> None:
    """Persist OHLCV ``df`` to disk merging with existing data.

    The resulting file contains a ``timestamp`` column in seconds with duplicate
    timestamps removed (keeping the latest entry).
    """

    if df is None or df.empty:
        return
    path = _build_path(storage_path, exchange_id, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = df.copy()
    new_df["timestamp"] = _to_seconds(new_df["timestamp"])

    existing = load_ohlcv(exchange_id, symbol, timeframe, storage_path)
    if existing is not None and not existing.empty:
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df = new_df.drop_duplicates(subset="timestamp", keep="last")
    new_df = new_df.sort_values("timestamp")
    new_df.to_parquet(path, index=False)

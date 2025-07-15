"""Helpers for loading Kraken minute data compressed in gzip format.

The functions here provide a simple way to read Kraken 1 minute OHLCV
history that is distributed as CSV files compressed with gzip.  Each
file follows Kraken's layout::

    timestamp,open,high,low,close,vwap,volume,count

``load_file`` reads a single file and converts the ``timestamp`` column
to :class:`~pandas.Timestamp` values.

``load_dir`` walks a directory tree, loads all ``.gz`` files and returns
their concatenated contents ordered by time.  Optional ``start`` and
``end`` parameters can be supplied to filter the resulting
:class:`~pandas.DataFrame`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

__all__ = ["load_file", "load_dir"]

COLUMNS = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]


def load_file(path: str | Path) -> pd.DataFrame:
    """Read a single Kraken CSV gzip file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.gz`` file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the OHLCV data with ``timestamp`` converted
        to ``datetime``.
    """

    df = pd.read_csv(path, compression="gzip", header=None, names=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_dir(directory: Path, start=None, end=None) -> pd.DataFrame:
    """Load all ``.gz`` files under ``directory`` and concatenate them.

    Files are discovered recursively and sorted by their timestamps.  If
    ``start`` or ``end`` are provided they are interpreted as datetimes
    and used to trim the resulting DataFrame.

    Parameters
    ----------
    directory:
        Root directory containing the gzip files.
    start, end:
        Optional datetime-like boundaries for filtering the result.

    Returns
    -------
    pandas.DataFrame
        Concatenated OHLCV data ordered by time.
    """

    directory = Path(directory)
    files = sorted(directory.rglob("*.gz"))

    frames: list[pd.DataFrame] = []
    for f in files:
        frames.append(load_file(f))

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("timestamp", inplace=True)

    if start is not None:
        start_ts = pd.to_datetime(start, utc=True)
        df = df[df["timestamp"] >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end, utc=True)
        df = df[df["timestamp"] <= end_ts]

    df.reset_index(drop=True, inplace=True)
    return df

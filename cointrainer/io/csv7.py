"""CSV7 reader stub used in tests."""

from __future__ import annotations

import io
import pandas as pd


def read_csv7(handle: io.StringIO | str) -> pd.DataFrame:
    """Parse a CSV7 stream into a DataFrame.

    The CSV7 format stores rows as ``ts,open,high,low,close,volume,trades``.
    This lightweight implementation is sufficient for unit tests and does not
    aim to be feature complete.
    """

    df = pd.read_csv(
        handle,
        names=["ts", "open", "high", "low", "close", "volume", "trades"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df.set_index("ts", inplace=True)
    return df


__all__ = ["read_csv7"]


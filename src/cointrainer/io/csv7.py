from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import IO, Union


def read_csv7(path_or_buf: Union[str, Path, IO[str]]) -> pd.DataFrame:
    """Read a CSV7-formatted file into a DataFrame.

    The expected columns are timestamp, open, high, low, close, volume, trades.
    The returned DataFrame is indexed by datetime named ``ts``.
    """
    df = pd.read_csv(
        path_or_buf,
        header=None,
        names=["ts", "open", "high", "low", "close", "volume", "trades"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df.set_index("ts", inplace=True)
    return df

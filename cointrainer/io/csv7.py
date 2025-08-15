from __future__ import annotations

import pandas as pd
from typing import IO, Iterable


COLUMNS = ["ts", "open", "high", "low", "close", "volume", "trades"]


def read_csv7(src: IO[str] | str | bytes | Iterable[str]):
    """Read a CSV7 stream into a DataFrame.

    CSV7 format: ts,open,high,low,close,volume,trades without header.
    The returned DataFrame has columns [open, high, low, close, volume, trades]
    and a datetime index named ``ts``.
    """
    df = pd.read_csv(src, header=None, names=COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df.set_index("ts", inplace=True)
    return df

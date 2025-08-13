from __future__ import annotations
from pathlib import Path
from typing import Union, Literal
import pandas as pd

CsvLike = Union[str, Path]


def read_csv7(path: CsvLike, *, tz: Literal["utc","naive"]="utc") -> pd.DataFrame:
    """
    Read a headerless 7-column CSV in the order:
      ts, open, high, low, close, volume, trades

    - ts is epoch seconds or milliseconds (auto-detected).
    - Returns OHLCV(+trades) with a UTC datetime index named 'ts'.
    """
    p = Path(path)
    df = pd.read_csv(p, header=None,
                     names=["ts", "open", "high", "low", "close", "volume", "trades"],
                     low_memory=False)
    # numeric coercion
    for c in ["ts", "open", "high", "low", "close", "volume", "trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).copy()

    # infer epoch unit (ms vs s)
    median_ts = df["ts"].median()
    unit = "ms" if median_ts > 1e12 else "s"

    dt = pd.to_datetime(df["ts"].astype("int64"), unit=unit, utc=True)
    if tz == "naive":
        dt = dt.tz_localize(None)

    df["ts"] = dt
    df = (df.sort_values("ts")
            .drop_duplicates("ts")
            .set_index("ts"))
    df.index.name = "ts"
    # enforce column order
    cols = ["open", "high", "low", "close", "volume", "trades"]
    return df[cols]

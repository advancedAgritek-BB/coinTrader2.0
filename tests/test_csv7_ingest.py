import io
import pandas as pd
import types, sys

# Provide a minimal stub for the ``cointrainer`` package
cointrainer = types.ModuleType("cointrainer")
io_mod = types.ModuleType("cointrainer.io")
csv7_mod = types.ModuleType("cointrainer.io.csv7")

def read_csv7(source):
    df = pd.read_csv(
        source,
        header=None,
        names=["ts", "open", "high", "low", "close", "volume", "trades"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.set_index("ts")

csv7_mod.read_csv7 = read_csv7
io_mod.csv7 = csv7_mod
cointrainer.io = io_mod
sys.modules.setdefault("cointrainer", cointrainer)
sys.modules.setdefault("cointrainer.io", io_mod)
sys.modules.setdefault("cointrainer.io.csv7", csv7_mod)

from cointrainer.io.csv7 import read_csv7


def test_read_csv7_basic():
    s = io.StringIO("1495122660,0.35,0.35,0.35,0.35,2.0,1\n1495122720,0.35,0.36,0.34,0.35,3.0,2\n")
    df = read_csv7(s)
    assert list(df.columns) == ["open","high","low","close","volume","trades"]
    assert df.index.name == "ts"
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df.index)

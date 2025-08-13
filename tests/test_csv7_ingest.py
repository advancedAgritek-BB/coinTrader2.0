import io
import pandas as pd
from cointrainer.io.csv7 import read_csv7


def test_read_csv7_basic():
    s = io.StringIO("1495122660,0.35,0.35,0.35,0.35,2.0,1\n1495122720,0.35,0.36,0.34,0.35,3.0,2\n")
    df = read_csv7(s)
    assert list(df.columns) == ["open","high","low","close","volume","trades"]
    assert df.index.name == "ts"
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df.index)

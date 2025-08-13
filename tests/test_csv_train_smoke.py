import io
import os
import pandas as pd
import numpy as np
import pytest

from cointrainer.train.local_csv import TrainConfig, train_from_csv7


@pytest.mark.skipif(
    pytest.importorskip("lightgbm", reason="LightGBM not installed") is None,
    reason="LightGBM not installed"
)
def test_csv_train_smoke(tmp_path):
    # tiny synthetic CSV7
    rows = []
    ts0 = 1_600_000_000
    price = 1.0
    for i in range(400):
        price *= (1 + 0.001 * np.sin(i/15))
        rows.append([ts0 + 60*i, price, price*1.001, price*0.999, price, 10+i%5, 1])

    s = io.StringIO("\n".join(",".join(str(x) for x in r) for r in rows))
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(s.getvalue())

    cfg = TrainConfig(symbol="TEST", horizon=10, hold=0.0005, n_estimators=80)
    model, meta = train_from_csv7(csv_path, cfg)
    assert "feature_list" in meta and len(meta["feature_list"]) >= 5

import pandas as pd
import numpy as np
import json
from pathlib import Path
from crypto_bot import torch_signal_model as tm


def _synthetic_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    data = {
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
        "volume": rng.rand(n) * 100,
    }
    df = pd.DataFrame(data)
    future_return = df["close"].shift(-5) / df["close"] - 1
    df["label"] = (future_return > 0).astype(int)
    return df


def test_training_and_prediction(tmp_path, monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(tm, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(tm, "REPORT_PATH", tmp_path / "report.json")
    model = tm.train_model(tm.extract_features(df), df.loc[tm.extract_features(df).index, "label"])
    assert tm.MODEL_PATH.exists()
    assert tm.REPORT_PATH.exists()
    if model is not None:
        pred = tm.predict_signal(df, model=model)
        assert 0.0 <= pred <= 1.0


def test_train_from_csv(tmp_path, monkeypatch):
    df = _synthetic_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(tm, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(tm, "REPORT_PATH", tmp_path / "report.json")
    model = tm.train_from_csv(csv)
    assert tm.MODEL_PATH.exists()
    assert tm.REPORT_PATH.exists()
    if model is not None:
        pred = tm.predict_signal(df, model=model)
        assert 0.0 <= pred <= 1.0

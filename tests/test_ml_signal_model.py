import pandas as pd
import numpy as np
from pathlib import Path
from crypto_bot import ml_signal_model as ml


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
    df["signal_strength"] = np.linspace(0, 1, n)
    return df


def test_training_and_prediction(tmp_path, monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.pkl")
    features = ml.extract_features(df)
    targets = df.loc[features.index, "signal_strength"]
    model = ml.train_model(features, targets)
    assert ml.MODEL_PATH.exists()
    pred = ml.predict_signal(df, model=model)
    assert 0.0 <= pred <= 1.0


def test_train_from_csv(tmp_path, monkeypatch):
    df = _synthetic_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.pkl")
    model = ml.train_from_csv(csv)
    assert ml.MODEL_PATH.exists()
    pred = ml.predict_signal(df, model=model)
    assert 0.0 <= pred <= 1.0

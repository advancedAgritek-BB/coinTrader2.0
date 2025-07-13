import json
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
    future_return = df["close"].shift(-5) / df["close"] - 1
    df["label"] = (future_return > 0).astype(int)
    return df


def test_training_and_prediction(tmp_path, monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(ml, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ml, "SCALER_PATH", tmp_path / "scaler.pkl")
    features = ml.extract_features(df)
    targets = df.loc[features.index, "label"]
    model = ml.train_model(features, targets)
    assert ml.MODEL_PATH.exists()
    assert ml.REPORT_PATH.exists()
    assert ml.SCALER_PATH.exists()
    report = json.loads(ml.REPORT_PATH.read_text())
    for key in ["accuracy", "auc", "precision", "recall", "trained_at"]:
        assert key in report
    pred = ml.predict_signal(df, model=model)
    assert 0.0 <= pred <= 1.0


def test_predict_single_path(tmp_path, monkeypatch):
    df = _synthetic_df()
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(ml, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ml, "SCALER_PATH", tmp_path / "scaler.pkl")

    features = ml.extract_features(df)
    targets = df.loc[features.index, "label"]
    model = ml.train_model(features, targets)

    call_counts = {"latest": 0, "extract": 0}

    original_extract = ml.extract_features

    def wrapped_extract(data, *a, **k):
        call_counts["extract"] += 1
        return original_extract(data, *a, **k)

    monkeypatch.setattr(ml, "extract_features", wrapped_extract)

    original_latest = ml.extract_latest_features

    def wrapped_latest(data):
        call_counts["latest"] += 1
        return original_latest(data)

    monkeypatch.setattr(ml, "extract_latest_features", wrapped_latest)

    pred = ml.predict_signal(df, model=model, scaler=ml.load_scaler())

    assert 0.0 <= pred <= 1.0
    assert call_counts["latest"] == 1
    assert call_counts["extract"] == 1


def test_train_from_csv(tmp_path, monkeypatch):
    df = _synthetic_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(ml, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ml, "SCALER_PATH", tmp_path / "scaler.pkl")
    model = ml.train_from_csv(csv)
    assert ml.MODEL_PATH.exists()
    assert ml.REPORT_PATH.exists()
    assert ml.SCALER_PATH.exists()
    report = json.loads(ml.REPORT_PATH.read_text())
    for key in ["accuracy", "auc", "precision", "recall", "trained_at"]:
        assert key in report
    pred = ml.predict_signal(df, model=model)
    assert 0.0 <= pred <= 1.0


def test_latest_features_consistency():
    df = _synthetic_df(200)
    full_last = ml.extract_features(df).iloc[[-1]]
    latest = ml.extract_latest_features(df, lookback=120)
    pd.testing.assert_frame_equal(
        full_last.reset_index(drop=True),
        latest.reset_index(drop=True),
        atol=1e-2,
    )


def test_extract_features_window():
    df = _synthetic_df(200)
    last_a = ml.extract_features(df).iloc[-1]
    last_b = ml.extract_features(df, window=len(df)).iloc[-1]
    pd.testing.assert_series_equal(last_a, last_b)


def test_order_book_imbalance_feature():
    df = _synthetic_df(60)
    df["bid_qty_0"] = 1.0
    df["bid_qty_1"] = 1.0
    df["ask_qty_0"] = 2.0
    df["ask_qty_1"] = 2.0
    feats = ml.extract_features(df)
    assert "order_book_imbalance" in feats.columns
    assert "order_book_imbalance_z" in feats.columns

import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "torch_price_model", ROOT / "crypto_bot" / "torch_price_model.py"
)
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def _df(n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
        "volume": rng.rand(n) * 100,
    })


def test_training_and_prediction(tmp_path, monkeypatch):
    df = _df()
    monkeypatch.setattr(tp, "MODEL_PATH", tmp_path / "model.pt")
    model = tp.train_model(df)
    assert tp.MODEL_PATH.exists()
    if model is not None:
        pred = tp.predict_price(df, model=model)
        assert isinstance(pred, float)

import pandas as pd
import numpy as np

from crypto_bot.models import torch_price_model as tpm


def _make_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
        "volume": rng.rand(n) * 100,
    })


def test_train_and_predict(tmp_path, monkeypatch):
    df = _make_df()
    cache = {"1h": {"AAA/USD": df}}
    monkeypatch.setattr(tpm, "MODEL_PATH", tmp_path / "price_model.pt")
    model = tpm.train_model(cache)
    if tpm.torch is not None:
        assert tpm.MODEL_PATH.exists()
        if model is not None:
            pred = tpm.predict_price(df, model=model)
            assert isinstance(pred, float)

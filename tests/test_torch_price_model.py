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
import asyncio

from crypto_bot import torch_price_model as tp
import crypto_bot.utils.market_analyzer as ma


def _df(n: int = 30) -> pd.DataFrame:

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


def test_training_and_prediction(tmp_path, monkeypatch):
    df = _df()
    monkeypatch.setattr(tp, "MODEL_PATH", tmp_path / "model.pt")
    model = tp.train_model(df)
    assert tp.MODEL_PATH.exists()
    if model is not None:
        pred = tp.predict_price(df, model=model)
        assert isinstance(pred, float)
    })


def test_predict_price(monkeypatch):
    df = _df()
    if tp.torch is not None and hasattr(tp.torch, "no_grad"):
        model = tp.PriceNet()
        pred = tp.predict_price(df, model=model)
        assert isinstance(pred, float)


def test_analyze_symbol_integration(monkeypatch):
    df = _df()
    df_map = {"1h": df}

    async def fake_classify(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_async", fake_classify)
    async def fake_cached(*_a, **_k):
        return "trending", {}
    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "detect_patterns", lambda *_a: {})
    monkeypatch.setattr(ma, "route", lambda *_a, **_k: lambda d, cfg=None: (0.5, "long"))

    async def fake_eval(*_a, **_k):
        return [(0.5, "long", None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: 0.1)
    monkeypatch.setattr(ma, "torch_predict_price", lambda _d: df["close"].iloc[-1] * 1.1)

    cfg = {"timeframe": "1h", "torch_price_model": {"enabled": True}}

    async def run():
        return await ma.analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["ai_pred_price"] == df["close"].iloc[-1] * 1.1
    assert res["direction"] == "long"
    assert res["score"] == 0.5 + abs(res["ai_pred_price"] - df["close"].iloc[-1]) / df["close"].iloc[-1]
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

import pandas as pd
import numpy as np
import asyncio

from crypto_bot import torch_price_model as tp
import crypto_bot.utils.market_analyzer as ma


def _df(n: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
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

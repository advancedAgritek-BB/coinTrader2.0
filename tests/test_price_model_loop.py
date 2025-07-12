import asyncio
import types
import pytest
import crypto_bot.main as main
import importlib.util
import pathlib
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "crypto_bot.volatility_filter", ROOT / "crypto_bot/volatility_filter.py"
)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)
sys.modules["crypto_bot.volatility_filter"] = vf

class DummyCtx:
    def __init__(self, interval=0.001):
        self.df_cache = {}
        self.config = {"torch_price_model": {"training_interval_hours": interval}}

class StopLoop(Exception):
    pass

def test_price_model_training_loop(monkeypatch):
    calls = {"train": 0, "sleep": 0}
    monkeypatch.setitem(
        sys.modules,
        "crypto_bot.torch_price_model",
        types.SimpleNamespace(train_model=lambda c: calls.__setitem__("train", calls["train"] + 1)),
    )

    async def fake_sleep(_):
        calls["sleep"] += 1
        raise StopLoop

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(main._price_model_training_loop(DummyCtx()))

    assert calls["train"] == 1

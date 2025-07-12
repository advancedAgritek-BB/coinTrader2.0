import asyncio
import numpy as np
import pandas as pd
import importlib.util
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load torch_price_model without importing the full package
spec_tp = importlib.util.spec_from_file_location(
    "crypto_bot.models.torch_price_model",
    ROOT / "crypto_bot" / "models" / "torch_price_model.py",
)
tpm = importlib.util.module_from_spec(spec_tp)
spec_tp.loader.exec_module(tpm)
sys.modules.setdefault("crypto_bot", types.ModuleType("crypto_bot"))
sys.modules.setdefault("crypto_bot.models", types.ModuleType("crypto_bot.models"))
sys.modules["crypto_bot.models.torch_price_model"] = tpm

# Create minimal stubs required for market_analyzer
for name in [
    "regime.pattern_detector",
    "regime.regime_classifier",
    "strategy_router",
    "meta_selector",
    "utils.perf",
    "signals.signal_scoring",
    "utils.rank_logger",
    "volatility_filter",
    "utils.logger",
    "utils",
    "utils.telegram",
    "utils.telemetry",
]:
    if "utils" in name and not name.startswith("utils."):
        sys.modules.setdefault("crypto_bot.utils", types.ModuleType("crypto_bot.utils"))
    module = types.ModuleType(f"crypto_bot.{name}")
    sys.modules[f"crypto_bot.{name}"] = module
    if name == "regime.pattern_detector":
        module.detect_patterns = lambda *_a: {}
    if name == "regime.regime_classifier":
        async def _classify(*_a, **_k):
            return "trending", {}

        module.classify_regime_async = _classify
        module.classify_regime_cached = _classify
    if name == "utils.perf":
        module.edge = lambda *_a, **_k: 1.0
    if name == "signals.signal_scoring":
        async def _eval(*a, **k):
            return [(0.5, "long", None)]

        module.evaluate_async = _eval
        module.evaluate_strategies = lambda *a, **k: []
    if name == "volatility_filter":
        module.calc_atr = lambda *_a, **_k: 0.1
    if name == "utils.rank_logger":
        module.log_second_place = lambda *a, **k: None
    if name == "strategy_router":
        module.RouterConfig = types.SimpleNamespace(from_dict=lambda d: types.SimpleNamespace(timeframe=d.get("timeframe", "1h")))
        module.route = lambda *_a, **_k: lambda d, cfg=None: (0.5, "long")
        module.strategy_name = lambda *_a: "dummy"
        module.get_strategies_for_regime = lambda *a, **k: []
        module.get_strategy_by_name = lambda *a, **k: None
        module.strategy_for = lambda *a, **k: lambda d, cfg=None: (0.5, "long")
        module.evaluate_regime = lambda *a, **k: {}
        module._build_mappings_cached = types.SimpleNamespace(cache_clear=lambda: None)
        module._CONFIG_REGISTRY = {}
    if name == "utils.telegram":
        module.TelegramNotifier = object

sys.modules["crypto_bot.utils.logger"].LOG_DIR = Path(".")
sys.modules["crypto_bot.utils.logger"].setup_logger = lambda *a, **k: None
sys.modules["crypto_bot.utils.telemetry"].telemetry = types.SimpleNamespace(inc=lambda *a, **k: None)
sys.modules.setdefault("crypto_bot.utils", types.ModuleType("crypto_bot.utils"))
sys.modules["crypto_bot.utils"].zscore = lambda s, w: s

spec_ma = importlib.util.spec_from_file_location(
    "crypto_bot.utils.market_analyzer",
    ROOT / "crypto_bot" / "utils" / "market_analyzer.py",
)
ma = importlib.util.module_from_spec(spec_ma)
sys.modules["crypto_bot.utils.market_analyzer"] = ma
spec_ma.loader.exec_module(ma)


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


def test_predict_price():
    df = _make_df()
    if tpm.torch is not None and hasattr(tpm.torch, "no_grad"):
        model = tpm.PriceNet()
        pred = tpm.predict_price(df, model=model)
        assert isinstance(pred, float)


def test_analyze_symbol_integration(monkeypatch):
    df = _make_df()
    df_map = {"1h": df}

    monkeypatch.setattr(ma, "torch_predict_price", lambda _d: df["close"].iloc[-1] * 1.1)

    cfg = {"timeframe": "1h", "torch_price_model": {"enabled": True}}

    async def run():
        return await ma.analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["ai_pred_price"] == df["close"].iloc[-1] * 1.1
    assert res["direction"] == "long"
    assert res["score"] == 0.5 + abs(res["ai_pred_price"] - df["close"].iloc[-1]) / df["close"].iloc[-1]

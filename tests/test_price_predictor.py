import asyncio
import numpy as np
import pandas as pd
import importlib.util
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load price_predictor without importing the full package
spec_pp = importlib.util.spec_from_file_location(
    "price_predictor", ROOT / "crypto_bot" / "models" / "price_predictor.py"
)
pp = importlib.util.module_from_spec(spec_pp)
spec_pp.loader.exec_module(pp)

# Create minimal stubs required for market_analyzer
sys.modules.setdefault("crypto_bot", types.ModuleType("crypto_bot"))
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
        base = sys.modules.setdefault("crypto_bot.utils", types.ModuleType("crypto_bot.utils"))
    module = types.ModuleType(f"crypto_bot.{name}")
    full_name = f"crypto_bot.{name}"
    sys.modules[full_name] = module
    if name == "regime.pattern_detector":
        module.detect_patterns = lambda *_a: {}
    if name == "regime.regime_classifier":
        module.classify_regime_async = lambda *_a, **_k: ("trending", {})
        module.classify_regime_cached = lambda *_a, **_k: ("trending", {})
    if name == "signals.signal_scoring":
        module.evaluate_async = lambda *a, **k: [(0.0, "none", None)]
        module.evaluate_strategies = lambda *a, **k: []
    if name == "volatility_filter":
        module.calc_atr = lambda *_a, **_k: 0.1
    if name == "utils.rank_logger":
        module.log_second_place = lambda *a, **k: None
    if name == "strategy_router":
        module.RouterConfig = types.SimpleNamespace(from_dict=lambda d: types.SimpleNamespace(timeframe=d.get("timeframe", "1h")))
        module.route = lambda *_a, **_k: lambda d, cfg=None: (0.0, "none")
        module.strategy_name = lambda *_a: "dummy"
        module.get_strategies_for_regime = lambda *a, **k: []
        module.get_strategy_by_name = lambda *a, **k: None
        module.strategy_for = lambda *a, **k: lambda d, cfg=None: (0.0, "none")
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


def _df(n: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
    })


def test_predict_score(monkeypatch):
    df = _df()
    if pp.torch is not None and hasattr(pp.torch, "no_grad"):
        class Dummy(pp.PriceNet):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True)
        dummy = Dummy()
        score = pp.predict_score(df, model=dummy)
        assert 0.0 <= score <= 1.0


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
    monkeypatch.setattr(pp, "predict_score", lambda _d, model=None: 0.7)

    cfg = {"timeframe": "1h", "ml_price_predictor": {"enabled": True}}

    async def run():
        return await ma.analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["price_score"] == 0.7


def test_analyze_symbol_direction_from_price(monkeypatch):
    df = _df()
    df_map = {"1h": df}

    async def fake_classify(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_async", fake_classify)

    async def fake_cached(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "detect_patterns", lambda *_a: {})
    monkeypatch.setattr(ma, "route", lambda *_a, **_k: lambda d, cfg=None: (0.0, "none"))

    async def fake_eval(*_a, **_k):
        return [(0.0, "none", None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: 0.1)
    monkeypatch.setattr(pp, "predict_score", lambda _d, model=None: 0.1)
    monkeypatch.setattr(pp, "predict_price", lambda _d, model=None: df["close"].iloc[-1] + 1)

    cfg = {"timeframe": "1h", "ml_price_predictor": {"enabled": True, "use_direction": True}}

    async def run():
        return await ma.analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["direction"] == "long"

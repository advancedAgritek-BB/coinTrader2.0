import sys
import types

from crypto_bot.strategies.loader import load_strategies


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(f"crypto_bot.strategy.{name}")
    class Strategy:
        def __init__(self):
            self.name = name
    mod.Strategy = Strategy
    return mod


def test_load_strategies_fallback(monkeypatch):
    # ensure fallback imports from crypto_bot.strategy works
    for name in ["grid_bot", "trend_bot", "micro_scalp_bot"]:
        mod = _make_module(name)
        monkeypatch.setitem(sys.modules, f"crypto_bot.strategy.{name}", mod)

    strategies = load_strategies("cex", ["grid_bot", "trend_bot", "micro_scalp_bot"])
    assert len(strategies) == 3
    assert sorted(s.name for s in strategies) == [
        "grid_bot",
        "micro_scalp_bot",
        "trend_bot",
    ]

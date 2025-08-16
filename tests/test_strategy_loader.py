import sys
import types

import pytest

from crypto_bot.strategies.loader import load_strategies


def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(f"crypto_bot.strategy.{name}")

    class Strategy:
        def __init__(self):
            self.name = name

    mod.Strategy = Strategy
    return mod


@pytest.fixture
def patch_strategy_modules(monkeypatch):
    def _patch(names):
        for name in names:
            mod = _make_module(name)
            monkeypatch.setitem(sys.modules, f"crypto_bot.strategy.{name}", mod)

    return _patch


def test_load_strategies_fallback(patch_strategy_modules):
    # ensure fallback imports from crypto_bot.strategy works
    patch_strategy_modules(["grid_bot", "trend_bot", "micro_scalp_bot"])
    patch_strategy_modules(["grid_bot", "trend_bot", "micro_scalp"])

    strategies = load_strategies("cex", ["grid_bot", "trend_bot", "micro_scalp"])

    patch_strategy_modules(["grid_bot", "trend_bot", "micro_scalp_bot"])
    strategies = load_strategies("cex", ["grid_bot", "trend_bot", "micro_scalp_bot"])

    assert len(strategies) == 3
    assert sorted(s.name for s in strategies) == [
        "grid_bot",
        "micro_scalp_bot",
        "trend_bot",
    ]

import types
import sys
from importlib import import_module


def test_range_arb_activation(monkeypatch):
    # Remove stubs for sklearn, scipy, and joblib inserted by test harness
    for key in [k for k in list(sys.modules) if k.startswith("sklearn") or k.startswith("scipy") or k.startswith("joblib")]:
        sys.modules.pop(key)
    import sklearn  # noqa: F401
    import scipy  # noqa: F401
    import joblib  # noqa: F401

    telegram_stub = types.ModuleType("crypto_bot.utils.telegram")
    class TelegramNotifier:  # minimal stub
        def __init__(self, *a, **k):
            pass
        async def notify_async(self, text):
            pass
        def notify(self, text):
            pass
    telegram_stub.TelegramNotifier = TelegramNotifier
    monkeypatch.setitem(sys.modules, "crypto_bot.utils.telegram", telegram_stub)

    sol_mempool_stub = types.ModuleType("crypto_bot.execution.solana_mempool")
    class SolanaMempoolMonitor:  # minimal stub
        pass
    sol_mempool_stub.SolanaMempoolMonitor = SolanaMempoolMonitor
    monkeypatch.setitem(
        sys.modules, "crypto_bot.execution.solana_mempool", sol_mempool_stub
    )

    sniper_stub = types.ModuleType("crypto_bot.strategies.sniper_solana")
    def generate_signal(df, config=None):
        return 0.0, "none"
    sniper_stub.generate_signal = generate_signal
    monkeypatch.setitem(
        sys.modules, "crypto_bot.strategies.sniper_solana", sniper_stub
    )

    # Reload strategy modules after cleaning stubs
    for key in [k for k in list(sys.modules) if k.startswith("crypto_bot.strategy") or k.startswith("crypto_bot.strategy_router")]:
        sys.modules.pop(key)

    strategy_router = import_module("crypto_bot.strategy_router")
    from crypto_bot.regime.regime_classifier import _ALL_REGIMES

    for regime in _ALL_REGIMES:
        strategies = strategy_router.get_strategies_for_regime(regime)
        assert any(
            getattr(fn, "__module__", "") == "crypto_bot.strategy.range_arb_bot"
            for fn in strategies
        )

import importlib
import sys
import types
from pathlib import Path


def _stub_strategy():
    mod = types.ModuleType("crypto_bot.strategy")
    mod.__path__ = [str((Path(__file__).resolve().parents[1] / "crypto_bot" / "strategy"))]
    mod.arbitrage_bot = types.SimpleNamespace()
    mod.high_freq_strategies = []
    sys.modules.setdefault("crypto_bot.strategy", mod)


_stub_strategy()
import crypto_bot.utils.pair_cache as pc


def test_load_liquid_pairs_empty(tmp_path, monkeypatch):
    file = tmp_path / "liquid_pairs.json"
    file.write_text("[]")
    monkeypatch.setattr(pc, "PAIR_FILE", file)
    assert pc.load_liquid_pairs() is None


def test_strategy_pair_fallback(tmp_path, monkeypatch):
    file = tmp_path / "liquid_pairs.json"
    file.write_text("[]")
    monkeypatch.setattr(pc, "PAIR_FILE", file)
    import crypto_bot.strategy.sniper_bot as sniper_bot
    import crypto_bot.strategy.dex_scalper as dex_scalper
    importlib.reload(sniper_bot)
    importlib.reload(dex_scalper)
    assert sniper_bot.ALLOWED_PAIRS == ["BTC/USD", "ETH/USD"]
    assert dex_scalper.ALLOWED_PAIRS == ["BTC/USD", "ETH/USD"]

import pandas as pd
import importlib.util
from pathlib import Path
import sys, types

sys.modules.setdefault("telegram", types.SimpleNamespace(Bot=None))
sys.modules.setdefault(
    "crypto_bot.strategy.sniper_bot", types.ModuleType("sniper_bot")
)
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault(
    "scipy.stats", types.SimpleNamespace(pearsonr=lambda x, y: (0.0, 0.0))
)

_spec = importlib.util.spec_from_file_location(
    "sniper_solana", Path(__file__).resolve().parents[1] / "crypto_bot/strategies/sniper_solana.py"
)
sniper_solana = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sniper_solana)


def make_df(prices):
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})


def test_skip_on_flags(monkeypatch):
    df = make_df([1.0, 1.0])
    score, direction = sniper_solana.generate_signal(df, config={"is_trading": False})
    assert score == 0.0
    assert direction == "none"

    score, direction = sniper_solana.generate_signal(df, config={"conf_pct": 0.6})
    assert score == 0.0
    assert direction == "none"


def test_uses_pyth_price(monkeypatch):
    df = make_df([1.0, 1.0])

    def fake_price(symbol, cfg=None):
        return 3.0

    monkeypatch.setattr(sniper_solana, "get_pyth_price", fake_price)
    cfg = {"token": "SOL", "atr_window": 1, "jump_mult": 1.0}
    score, direction = sniper_solana.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score == 1.0

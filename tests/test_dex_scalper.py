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
    "dex_scalper", Path(__file__).resolve().parents[1] / "crypto_bot/strategy/dex_scalper.py"
)
dex_scalper = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(dex_scalper)


def test_scalper_long_signal():
    close = pd.Series(range(1, 21))
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'long'
    assert 0 < score <= 1


def test_scalper_short_signal():
    close = pd.Series(range(40, 0, -1))
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'short'
    assert 0 < score <= 1


def test_scalper_neutral_signal():
    close = pd.Series([100.0] * 30)
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'none'
    assert score == 0.0


def test_scalper_min_data():
    close = pd.Series([100.0] * 5)
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'none'
    assert score == 0.0


def test_accepts_short_history():
    close = pd.Series(range(1, 11))
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'long'
    assert score > 0


def test_scalper_custom_config():
    close = pd.Series(range(1, 41))
    df = pd.DataFrame({'close': close})
    cfg = {'dex_scalper': {'ema_fast': 3, 'ema_slow': 10, 'min_signal_score': 0.05}}
    score, direction = dex_scalper.generate_signal(df, config=cfg)
    assert direction == 'long'
    assert score > 0


def test_priority_fee_aborts(monkeypatch):
    close = pd.Series(range(1, 21))
    df = pd.DataFrame({"close": close})
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "50")
    cfg = {"dex_scalper": {"priority_fee_cap_micro_lamports": 10}}
    score, direction = dex_scalper.generate_signal(df, config=cfg)
    assert score == 0.0
    assert direction == "none"


def test_priority_fee_below_threshold(monkeypatch):
    close = pd.Series(range(1, 21))
    df = pd.DataFrame({"close": close})
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "5")
    cfg = {"dex_scalper": {"priority_fee_cap_micro_lamports": 10}}
    score, direction = dex_scalper.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0


class DummyMonitor:
    def __init__(self, fee):
        self._fee = fee

    async def fetch_priority_fee(self):
        return self._fee


def test_monitor_fee_blocks_signal():
    df = pd.DataFrame({"close": pd.Series(range(1, 21))})
    cfg = {"dex_scalper": {"priority_fee_cap_micro_lamports": 10}}
    score, direction = dex_scalper.generate_signal(
        df, config=cfg, mempool_monitor=DummyMonitor(15)
    )
    assert score == 0.0
    assert direction == "none"


def test_monitor_fee_allows_signal():
    df = pd.DataFrame({"close": pd.Series(range(1, 21))})
    cfg = {"dex_scalper": {"priority_fee_cap_micro_lamports": 10}}
    score, direction = dex_scalper.generate_signal(
        df, config=cfg, mempool_monitor=DummyMonitor(5)
    )
    assert direction == "long"
    assert score > 0

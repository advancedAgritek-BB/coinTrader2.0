import importlib
import sys
import types

import pandas as pd
import pytest
from crypto_bot import cooldown_manager

sys.modules.setdefault(
    "crypto_bot.strategy.sniper_bot", types.ModuleType("crypto_bot.strategy.sniper_bot")
)
sys.modules.setdefault(
    "crypto_bot.strategy.sniper_solana", types.ModuleType("crypto_bot.strategy.sniper_solana")
)

dip_hunter = importlib.import_module("crypto_bot.strategy.dip_hunter")


def _df_dip(size: int = 150) -> pd.DataFrame:
    close = [100.0] * (size - 3) + [98.0, 97.0, 95.0]
    data = {
        "open": close,
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [100.0] * (size - 1) + [200.0],
    }
    return pd.DataFrame(data)


def _patch_indicators(monkeypatch, length: int):
    monkeypatch.setattr(
        dip_hunter.ta.momentum,
        "rsi",
        lambda series, window=14: pd.Series([20.0] * len(series), index=series.index),
    )

    class DummyBB:
        def __init__(self, close, window=20):
            self.close = close

        def bollinger_pband(self):
            return pd.Series([-0.2] * len(self.close), index=self.close.index)

    monkeypatch.setattr(dip_hunter.ta.volatility, "BollingerBands", DummyBB)

    class DummyADX:
        def __init__(self, *args, **kwargs):
            self.length = length

        def adx(self):
            return pd.Series([10.0] * self.length)

    monkeypatch.setattr(dip_hunter, "ADXIndicator", DummyADX)
    monkeypatch.setattr(
        dip_hunter.stats,
        "zscore",
        lambda s, lookback=20: pd.Series([2.0] * len(s), index=s.index),
    )


def test_long_signal(monkeypatch):
    df = _df_dip()
    _patch_indicators(monkeypatch, len(df))
    score, direction = dip_hunter.generate_signal(df, config={"symbol": "BTC/USD"})
    assert direction == "long"
    assert score > 0


def test_generate_signal_without_higher_df(monkeypatch):
    df = _df_dip()
    _patch_indicators(monkeypatch, len(df))
    result = dip_hunter.generate_signal(df, higher_df=None, config={"symbol": "BTC/USD"})
    assert isinstance(result, tuple)
    score, direction = result
    assert direction == "long"


def test_cooldown_blocks(monkeypatch):
    df = _df_dip()
    _patch_indicators(monkeypatch, len(df))
    cooldown_manager.configure(10)
    cooldown_manager.cooldowns.clear()
    cfg = {"symbol": "BTC/USD", "dip_hunter": {"cooldown_enabled": True}}

    score, direction = dip_hunter.generate_signal(df, config=cfg)
    assert direction == "long" and score > 0

    score2, direction2 = dip_hunter.generate_signal(df, config=cfg)
    assert direction2 == "none" and score2 == 0.0


@pytest.mark.parametrize("rows", [15, 104])
def test_handles_small_frames(monkeypatch, rows):
    df = _df_dip(rows)
    _patch_indicators(monkeypatch, len(df))
    result = dip_hunter.generate_signal(df, config={"symbol": "BTC/USD"})
    assert result == (0.0, "none")

import importlib
import types
import pandas as pd
import numpy as np
import pytest

from crypto_bot.core.pipeline import scoring_loop
from crypto_bot.core.queues import trade_queue

mean_bot = importlib.import_module("crypto_bot.strategy.mean_bot")
momentum_bot = importlib.import_module("crypto_bot.strategy.momentum_bot")


def _dummy_df(length: int = 30) -> pd.DataFrame:
    prices = list(range(length))
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [100] * length,
        }
    )
@pytest.mark.asyncio
async def test_mean_bot_enqueues_trade(monkeypatch):
    """Pipeline should enqueue trades when mean_bot signals."""

    df = _dummy_df()
    monkeypatch.setattr(
        mean_bot,
        "generate_signal",
        lambda *a, **k: (0.5, "long"),
    )
    strategy = types.SimpleNamespace(name="mean_bot", generate_signal=mean_bot.generate_signal)
    config = {"mean_bot": {}, "thresholds": {"mean_bot": {"1h": 0.1}}}
    while not trade_queue.empty():
        trade_queue.get_nowait()
        trade_queue.task_done()
    await scoring_loop(config, strategy, "BTC/USD", "1h", df)
    cand = trade_queue.get_nowait()
    assert cand["strategy"] == "mean_bot"
    assert cand["side"] == "long"


@pytest.mark.asyncio
async def test_momentum_bot_enqueues_trade(monkeypatch):
    df = _dummy_df()
    monkeypatch.setattr(
        momentum_bot.ta.trend,
        "macd",
        lambda *a, **k: pd.Series([1] * len(df)),
    )
    monkeypatch.setattr(
        momentum_bot.ta.momentum,
        "rsi",
        lambda *a, **k: pd.Series([30] * len(df)),
    )
    strategy = types.SimpleNamespace(name="momentum_bot", generate_signal=momentum_bot.generate_signal)
    config = {
        "momentum_bot": {"atr_normalization": False},
        "thresholds": {"momentum_bot": {"1h": 0.5}},
    }
    while not trade_queue.empty():
        trade_queue.get_nowait()
        trade_queue.task_done()
    await scoring_loop(config, strategy, "BTC/USD", "1h", df)
    cand = trade_queue.get_nowait()
    assert cand["strategy"] == "momentum_bot"
    assert cand["side"] == "long"

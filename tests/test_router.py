import asyncio
import time

import pytest

from crypto_bot import strategy_router

def dummy_signal(df, cfg=None):
    return 0.5, "long"

@pytest.mark.asyncio
async def test_symbol_lock_single_active(monkeypatch):
    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda n: dummy_signal if n == "dummy" else None,
    )
    cfg = {"strategy_router": {"regimes": {"trending": ["dummy"]}}}
    fn = strategy_router.route("trending", "cex", cfg)

    times = {}

    async def caller(name):
        start = time.perf_counter()
        await fn(None, {"symbol": "BTC/USD"})
        acquired = time.perf_counter()
        times[name] = (start, acquired)
        await asyncio.sleep(0.05)
        await strategy_router.release_symbol_lock("BTC/USD")

    t1 = asyncio.create_task(caller("a"))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(caller("b"))
    await asyncio.gather(t1, t2)

    assert times["a"][1] < times["b"][1]
import pandas as pd
import logging

import crypto_bot.strategy_router as router
from crypto_bot.signals.signal_scoring import evaluate_async


def slow_strategy(df, cfg=None):
    import time
    time.sleep(6)
    return 1.0, "long"


def test_strategy_timeout_logged(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(router, "get_strategy_by_name", lambda n: slow_strategy)

    cfg = {"strategy_router": {"regimes": {"trending": ["slow"]}}}
    fn = router.route("trending", "cex", cfg)

    async def run():
        df = pd.DataFrame({"close": [1, 2]})
        return await evaluate_async([fn], df, {})

    res = asyncio.run(run())
    assert res == [(0.0, "none", None)]
    assert any("TIMEOUT" in r.getMessage() for r in caplog.records)
import pytest
import crypto_bot.strategy_router as sr
from crypto_bot.strategy_router import RouterConfig, BotStats


def make_bot(name):
    def bot(df, cfg=None):
        return 0.0, "long"
    bot.__name__ = name
    return bot


def test_higher_score_selected(monkeypatch):
    bot_a = make_bot("bot_a")
    bot_b = make_bot("bot_b")
    monkeypatch.setattr(sr, "get_strategy_by_name", lambda n: {"bot_a": bot_a, "bot_b": bot_b}.get(n))
    def fake_stats(name):
        if name == "bot_a":
            return BotStats(sharpe_30d=1.0, win_rate_30d=0.5, avg_r_multiple=0.5)
        return BotStats()
    monkeypatch.setattr(sr, "load_bot_stats", fake_stats)
    cfg = RouterConfig.from_dict({"strategy_router": {"regimes": {"trending": ["bot_b", "bot_a"]}}})
    result = sr.get_strategies_for_regime("trending", cfg)
    assert result[0] is bot_a
    assert result[1] is bot_b


def test_equal_scores_use_order(monkeypatch):
    bot_a = make_bot("bot_a")
    bot_b = make_bot("bot_b")
    monkeypatch.setattr(sr, "get_strategy_by_name", lambda n: {"bot_a": bot_a, "bot_b": bot_b}.get(n))
    monkeypatch.setattr(sr, "load_bot_stats", lambda n: BotStats())
    cfg = RouterConfig.from_dict({"strategy_router": {"regimes": {"trending": ["bot_a", "bot_b"]}}})
    result = sr.get_strategies_for_regime("trending", cfg)
    assert result == [bot_a, bot_b]


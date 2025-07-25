import asyncio
from collections import deque
import pandas as pd
import sys
import types
from crypto_bot.phase_runner import BotContext

# Stub ccxt to avoid heavy optional dependency during import
sys.modules.setdefault(
    "ccxt", types.SimpleNamespace(async_support=types.SimpleNamespace())
)
sys.modules.setdefault("ccxt.async_support", types.SimpleNamespace())

import crypto_bot.main as main


async def dummy_update(*args, **kwargs):
    dummy_update.kwargs = kwargs
    return {}


def test_update_caches_default_limit(monkeypatch):
    ctx = BotContext(
        positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h"}
    )
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", dummy_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update)
    asyncio.run(main.update_caches(ctx))
    assert dummy_update.kwargs["limit"] == 150


def test_update_caches_override(monkeypatch):
    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config={"timeframe": "1h", "cycle_lookback_limit": 50},
    )
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", dummy_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update)
    asyncio.run(main.update_caches(ctx))
    assert dummy_update.kwargs["limit"] == 50


def test_update_caches_volume_priority(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1000000],
        }
    )

    async def fake_update(*args, **kwargs):
        return {"1h": {"FOO/USDC": df}}

    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config={"timeframe": "1h", "bounce_scalper": {"vol_zscore_threshold": 1.0}},
    )
    ctx.exchange = object()
    ctx.current_batch = ["FOO/USDC"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", fake_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())

    asyncio.run(main.update_caches(ctx))

    assert list(main.symbol_priority_queue) == ["FOO/USDC"]


async def dummy_update_multi(*args, **kwargs):
    dummy_update_multi.kwargs = kwargs
    return {}


async def dummy_update_regime(*args, **kwargs):
    dummy_update_regime.kwargs = kwargs
    return {}


def test_update_caches_volatility_adjusts_concurrency(monkeypatch):
    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config={"timeframe": "1h", "max_concurrent_ohlcv": 8},
    )
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    ctx.volatility_factor = 6.0
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", dummy_update_multi)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update_regime)
    asyncio.run(main.update_caches(ctx))
    assert dummy_update_multi.kwargs["max_concurrent"] == 4
    assert dummy_update_regime.kwargs["max_concurrent"] == 4


async def success_update(*args, **kwargs):
    success_update.called = True
    return {}


async def fail_then_succeed(*args, **kwargs):
    fail_then_succeed.calls += 1
    if fail_then_succeed.calls == 1:
        raise Exception("fail")
    return {}


async def record_regime(*args, **kwargs):
    record_regime.called = True
    return {}


def test_update_caches_regime_called_normal(monkeypatch):
    ctx = BotContext(
        positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h"}
    )
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", success_update)
    record_regime.called = False
    monkeypatch.setattr(main, "update_regime_tf_cache", record_regime)
    asyncio.run(main.update_caches(ctx))
    assert getattr(record_regime, "called", False)


def test_update_caches_regime_called_on_fallback(monkeypatch):
    ctx = BotContext(
        positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h"}
    )
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    fail_then_succeed.calls = 0
    record_regime.called = False
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", fail_then_succeed)
    monkeypatch.setattr(main, "update_regime_tf_cache", record_regime)
    asyncio.run(main.update_caches(ctx))
    assert getattr(record_regime, "called", False)

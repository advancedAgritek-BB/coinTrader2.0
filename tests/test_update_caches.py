import asyncio
from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main


async def dummy_update(*args, **kwargs):
    dummy_update.kwargs = kwargs
    return {}


def test_update_caches_default_limit(monkeypatch):
    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h"})
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", dummy_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update)
    asyncio.run(main.update_caches(ctx))
    assert dummy_update.kwargs["limit"] == 120


def test_update_caches_override(monkeypatch):
    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h", "cycle_lookback_limit": 50})
    ctx.exchange = object()
    ctx.current_batch = ["BTC/USDT"]
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", dummy_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", dummy_update)
    asyncio.run(main.update_caches(ctx))
    assert dummy_update.kwargs["limit"] == 50


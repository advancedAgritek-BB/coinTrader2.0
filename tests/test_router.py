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

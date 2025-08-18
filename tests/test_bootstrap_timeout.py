import asyncio
from types import SimpleNamespace

import pytest

import crypto_bot.main as main


@pytest.mark.asyncio
async def test_bootstrap_timeout(monkeypatch):
    paused = asyncio.Event()
    resumed = asyncio.Event()
    cycle_count = 0

    async def slow_update(*args, **kwargs):
        slow_update.calls += 1
        if slow_update.calls == 1:
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                paused.set()
                async def _resume():
                    await asyncio.sleep(0.2)
                    resumed.set()
                asyncio.create_task(_resume())
                raise
        return {}

    slow_update.calls = 0
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", slow_update)

    async def run_cycle():
        nonlocal cycle_count
        cycle_count += 1
        await main.update_multi_tf_ohlcv_cache(None, {}, [], {})

    ctx = SimpleNamespace()
    config = {"loop_interval_minutes": 0, "evaluation_timeout": 0.05}
    stop_reason = ["completed"]

    loop_task = asyncio.create_task(
        main.evaluation_loop(run_cycle, ctx, config, stop_reason)
    )

    await asyncio.sleep(0.1)
    assert paused.is_set()
    assert not resumed.is_set()
    assert cycle_count >= 2

    await asyncio.wait_for(resumed.wait(), 1)

    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

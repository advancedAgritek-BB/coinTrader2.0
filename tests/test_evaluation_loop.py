import asyncio
import logging
from types import SimpleNamespace

import pytest

import crypto_bot.main as main


@pytest.mark.asyncio
async def test_evaluation_loop_timeout_logs(caplog):
    caplog.set_level(logging.INFO)

    async def slow_cycle():
        await asyncio.Event().wait()

    ctx = SimpleNamespace(active_universe=[], current_batch=[])
    config = {"loop_interval_minutes": 0, "evaluation_timeout": 0.01}
    stop_reason = ["completed"]

    loop_task = asyncio.create_task(
        main.evaluation_loop(slow_cycle, ctx, config, stop_reason)
    )
    await asyncio.sleep(0.05)
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

    messages = [r.getMessage() for r in caplog.records]
    assert any("Starting evaluation cycle" in m for m in messages)
    assert any("timed out" in m for m in messages)
    assert any("Active universe" in m for m in messages)

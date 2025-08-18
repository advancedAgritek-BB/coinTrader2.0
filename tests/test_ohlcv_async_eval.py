import asyncio
import contextlib
import time
import pandas as pd
import pytest

from crypto_bot.utils import market_loader
from crypto_bot.utils.market_loader import update_ohlcv_cache


@pytest.mark.asyncio
async def test_evaluation_runs_during_ohlcv_processing(monkeypatch):
    class DummyExchange:
        id = "dummy"
        markets = {"AAA/USD": {}}

        async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
            start = since or int(time.time() * 1000)
            return [
                [start + i * 3600 * 1000, 1, 1, 1, 1, 1]
                for i in range(limit)
            ]

    # Simulate expensive pandas operations that would block the event loop
    # if not executed in a worker thread.
    orig_resample = pd.DataFrame.resample
    orig_concat = pd.concat

    def slow_resample(self, *args, **kwargs):
        time.sleep(0.2)
        return orig_resample(self, *args, **kwargs)

    def slow_concat(*args, **kwargs):
        time.sleep(0.2)
        return orig_concat(*args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "resample", slow_resample)
    monkeypatch.setattr(pd, "concat", slow_concat)

    cache: dict[str, pd.DataFrame] = {}
    symbols = ["AAA/USD"]

    eval_flag = False

    async def evaluator():
        nonlocal eval_flag
        await asyncio.sleep(0.05)
        eval_flag = True

    await asyncio.gather(
        update_ohlcv_cache(DummyExchange(), cache, symbols, timeframe="1h", limit=10),
        evaluator(),
    )

    # Ensure any background worker tasks are cleaned up for the test
    for task in list(market_loader._OHLCV_BATCH_TASKS.values()):
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    assert eval_flag, "evaluation task did not run while OHLCV processing executed"


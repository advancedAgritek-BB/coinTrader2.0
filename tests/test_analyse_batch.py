import asyncio
import types

import pytest

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main


class Counter:
    def __init__(self):
        self.active = 0
        self.max_active = 0


def test_analyse_batch_semaphore(monkeypatch):
    counter = Counter()
    sem = asyncio.Semaphore(2)

    async def dummy_analyze_symbol(*args, **kwargs):
        async with sem:
            counter.active += 1
            counter.max_active = max(counter.max_active, counter.active)
            await asyncio.sleep(0.01)
            counter.active -= 1
        return {"symbol": args[0], "df": {}, "score": 0.0, "direction": "none"}

    monkeypatch.setattr(main, "analyze_symbol", dummy_analyze_symbol)

    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={})
    ctx.current_batch = list("ABCDE")

    asyncio.run(main.analyse_batch(ctx))

    assert counter.max_active <= 2

import yaml
from crypto_bot.utils import market_loader


class CountingExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return [[0] * 6]


def test_update_ohlcv_cache_default_max_concurrent(tmp_path, monkeypatch):
    cfg = {"max_concurrent_ohlcv": 3}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    monkeypatch.setattr(market_loader, "CONFIG_PATH", path)
    market_loader.SEMA = None
    market_loader.configure(max_concurrent=None)

    ex = CountingExchange()
    cache = {}
    symbols = ["A", "B", "C", "D"]

    asyncio.run(market_loader.update_ohlcv_cache(ex, cache, symbols))

    assert ex.max_active <= 3

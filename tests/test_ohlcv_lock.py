import asyncio
import logging

from crypto_bot.utils import market_loader


def test_skip_overlapping_tf_updates(monkeypatch, caplog):
    market_loader._TF_LOCKS.clear()

    async def slow_listing(_sym):
        await asyncio.sleep(0.2)
        return None

    async def dummy_update(*_a, **_k):
        return {}

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", slow_listing)
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", dummy_update)

    class DummyEx:
        id = "kraken"
        timeframes = {"1h": "1h"}

    async def runner():
        async def first():
            await market_loader.update_multi_tf_ohlcv_cache(
                DummyEx(), {}, ["BTC/USD"], {"timeframes": ["1h"]}
            )

        async def second():
            await asyncio.sleep(0.05)
            await market_loader.update_multi_tf_ohlcv_cache(
                DummyEx(), {}, ["BTC/USD"], {"timeframes": ["1h"]}
            )

        await asyncio.gather(first(), second())

    with caplog.at_level(logging.INFO):
        asyncio.run(runner())

    assert any(
        "Starting update for timeframe 1h" in r.getMessage() for r in caplog.records
    )
    assert any(
        "Skip: 1h update already running." in r.getMessage() for r in caplog.records
    )

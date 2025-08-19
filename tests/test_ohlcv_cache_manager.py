import asyncio
import logging
from types import SimpleNamespace

from crypto_bot.data.ohlcv_cache import OHLCVCache


class DummyCache(OHLCVCache):
    async def _fetch_and_store(self, tf, warmup=None, start=None):
        # Simulate a quick fetch
        await asyncio.sleep(0)


def test_update_intraday_runs_all_timeframes(caplog):
    cfg = SimpleNamespace(
        warmup_candles={'1m': 1000, '5m': 600},
        backfill_days={'1m': 2, '5m': 3},
        deep_backfill_days={'1m': 2, '5m': 3},
    )
    cache = DummyCache(cfg, logger=logging.getLogger('test'))
    caplog.set_level('INFO')
    asyncio.run(cache.update_intraday(['1m', '5m']))
    assert 'Starting OHLCV update for timeframe 1m' in caplog.text
    assert 'Starting OHLCV update for timeframe 5m' in caplog.text

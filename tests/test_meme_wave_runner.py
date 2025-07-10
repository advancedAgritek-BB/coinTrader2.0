import asyncio
import pytest

from crypto_bot.solana import meme_wave_runner
from crypto_bot.solana.watcher import NewPoolEvent, PoolWatcher
from crypto_bot.main import SessionState, initial_scan


class DummyExchange:
    pass


@pytest.mark.asyncio
async def test_watch_starts_before_initial_scan(monkeypatch):
    started = asyncio.Event()

    async def watch_stub(self):
        started.set()
        yield NewPoolEvent("P", "M", "C", 0.0)

    monkeypatch.setattr(PoolWatcher, "watch", watch_stub)

    async def fake_scan(exchange, cfg, state, notifier=None):
        await asyncio.sleep(0.01)
        assert started.is_set()


    monkeypatch.setattr("crypto_bot.main.update_multi_tf_ohlcv_cache", lambda *a, **k: {})
    monkeypatch.setattr("crypto_bot.main.update_regime_tf_cache", lambda *a, **k: {})
    monkeypatch.setattr("crypto_bot.main.initial_scan", fake_scan)

    task = meme_wave_runner.start_runner({"enabled": True, "pool": {"url": "u", "interval": 0.001}})
    assert isinstance(task, asyncio.Task)
    await initial_scan(DummyExchange(), {"symbols": ["BTC/USDT"], "timeframes": ["1h"]}, SessionState())
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


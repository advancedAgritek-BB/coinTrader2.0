import asyncio
import pytest

pytestmark = pytest.mark.asyncio

from crypto_bot.solana import meme_wave_runner
from crypto_bot.solana.watcher import NewPoolEvent, PoolWatcher
from crypto_bot.main import SessionState, initial_scan
from crypto_bot.phase_runner import BotContext, PhaseRunner


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
    await initial_scan(DummyExchange(), {"symbols": ["XBT/USDT"], "timeframes": ["1h"]}, SessionState())
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_trading_phases_start_before_scan_finishes(monkeypatch):
    started = asyncio.Event()

    async def slow_scan(exchange, cfg, state, notifier=None):
        await asyncio.sleep(0.01)
        assert started.is_set()

    monkeypatch.setattr(
        "crypto_bot.main.update_multi_tf_ohlcv_cache", lambda *a, **k: {}
    )
    monkeypatch.setattr("crypto_bot.main.update_regime_tf_cache", lambda *a, **k: {})

    state = SessionState()
    config = {"symbols": ["XBT/USDT"], "timeframes": ["1h"], "scan_in_background": True}

    state.scan_task = asyncio.create_task(slow_scan(DummyExchange(), config, state))

    async def fake_phase(ctx):
        started.set()

    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config=config)
    await PhaseRunner([fake_phase]).run(ctx)

    await state.scan_task


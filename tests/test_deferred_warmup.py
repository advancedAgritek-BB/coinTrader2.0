import asyncio
import pytest

from crypto_bot.main import initial_scan, SessionState


class DummyExchange:
    pass


@pytest.mark.asyncio
async def test_deferred_warmup(monkeypatch):
    calls: list[list[str]] = []
    event = asyncio.Event()
    tasks: list[asyncio.Task] = []

    async def fake_update_multi(exchange, cache, batch, cfg, **kwargs):
        tfs = cfg.get("timeframes")
        if "5m" in tfs and "1h" not in tfs:
            await event.wait()
        calls.append(tfs)
        return {}

    def fake_register_task(task: asyncio.Task | None):
        if task:
            tasks.append(task)
        return task

    monkeypatch.setattr("crypto_bot.main.update_multi_tf_ohlcv_cache", fake_update_multi)

    async def fake_update_regime(*a, **k):
        return {}

    monkeypatch.setattr("crypto_bot.main.update_regime_tf_cache", fake_update_regime)
    monkeypatch.setattr("crypto_bot.main.register_task", fake_register_task)
    async def fake_get_filtered_symbols(ex, cfg):
        return ([(cfg["symbols"][0], 0.0)], cfg.get("onchain_symbols", []))
    monkeypatch.setattr(
        "crypto_bot.main.get_filtered_symbols",
        fake_get_filtered_symbols,
    )

    cfg = {
        "symbols": ["BTC/USD"],
        "timeframes": ["1h", "5m"],
        "scan_lookback_limit": 50,
        "ohlcv": {
            "bootstrap_timeframes": ["1h"],
            "defer_timeframes": ["5m"],
        },
    }

    await initial_scan(DummyExchange(), cfg, SessionState())

    assert calls[0] == ["1h"]
    assert tasks and not tasks[0].done()

    event.set()
    await asyncio.gather(*tasks)
    assert len(calls) == 2
    assert "5m" in calls[1]

import asyncio
import pytest

import crypto_bot.solana.runner as runner
from crypto_bot.solana.watcher import NewPoolEvent


class DummyWatcher:
    def __init__(self, *a, **k):
        pass

    async def watch(self):
        yield NewPoolEvent("P", "M", "C", 1.0, 2)

    def stop(self):
        pass

    def _predict_breakout(self, event):
        return 0.8


@pytest.mark.asyncio
async def test_run_triggers_cross_chain(monkeypatch):
    monkeypatch.setattr(runner, "PoolWatcher", DummyWatcher)
    monkeypatch.setattr(runner, "score_event", lambda *a, **k: 0.8)

    async def dummy_snipe(event, score, cfg):
        return {}

    monkeypatch.setattr(runner.executor, "snipe", dummy_snipe)

    called = {}

    async def dummy_trade(exchange, ws_client, symbol, side, amount, **kwargs):
        called["args"] = (exchange, ws_client, symbol, side, amount)
        called["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(runner, "cross_chain_trade", dummy_trade)

    exchange = object()
    ws_client = object()
    cfg = {
        "pool": {},
        "scoring": {},
        "execution": {},
        "arbitrage": {
            "exchange": exchange,
            "ws_client": ws_client,
            "symbol": "SOL/USDC",
            "side": "buy",
            "amount": 1,
            "dry_run": False,
            "slippage_bps": 25,
            "use_websocket": True,
        },
    }

    await runner.run(cfg)
    assert called["args"] == (exchange, ws_client, "SOL/USDC", "buy", 1.0)


class ErrorWatcher:
    def __init__(self, *a, **k):
        pass

    async def watch(self):
        raise RuntimeError("boom")
        yield  # pragma: no cover - async generator form

    def stop(self):
        pass

    def _predict_breakout(self, event):
        return 0.8


@pytest.mark.asyncio
async def test_fallback_called_on_watch_error(monkeypatch):
    monkeypatch.setattr(runner, "PoolWatcher", ErrorWatcher)
    called = False

    async def dummy_poll(cfg):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "poll_fallback", dummy_poll)

    await runner.run({"pool": {}})
    assert called

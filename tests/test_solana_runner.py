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
    called = False
    async def dummy_trade(*a, **k):
        nonlocal called
        called = True
        return {}
    monkeypatch.setattr(runner, "cross_chain_trade", dummy_trade)

    cfg = {
        "pool": {},
        "scoring": {},
        "execution": {},
        "arbitrage": {"symbol": "SOL/USDC", "side": "buy", "amount": 1},
    }
    await runner.run(cfg)
    assert called

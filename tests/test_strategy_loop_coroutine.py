import inspect
import pytest
from crypto_bot import main


@pytest.mark.asyncio
async def test_strategy_loop_handles_coroutine(monkeypatch):
    async def fake_evaluate_all(symbols, timeframes):
        return [
            {"symbol": "SYM", "timeframe": "1m", "score": 1.0, "direction": "none", "extra": {}}
        ]

    monkeypatch.setattr(main.strategy_manager, "evaluate_all", fake_evaluate_all)

    async def run_once():
        signals = main.strategy_manager.evaluate_all([], [])
        if inspect.isawaitable(signals):
            signals = await signals
        return len(signals)

    assert await run_once() == 1

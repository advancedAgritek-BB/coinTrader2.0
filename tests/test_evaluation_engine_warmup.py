import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from crypto_bot.engine.evaluation_engine import StreamEvaluationEngine
import crypto_bot.engine.evaluation_engine as evaluation_engine


@pytest.mark.asyncio
async def test_symbols_evaluated_when_warmup_disabled(monkeypatch):
    called: list[str] = []

    async def eval_fn(symbol: str, ctx: dict) -> None:
        called.append(symbol)

    data = SimpleNamespace(ready=lambda symbol, tf: False)
    cfg = SimpleNamespace(
        trading=SimpleNamespace(mode="auto"),
        evaluation=SimpleNamespace(workers=1),
        router=SimpleNamespace(require_warm=False),
    )

    monkeypatch.setattr(
        evaluation_engine,
        "load_strategies",
        lambda mode, return_errors=True: ([object()], {}),
    )

    engine = StreamEvaluationEngine(eval_fn, cfg=cfg, data=data)
    await engine.start()
    await engine.enqueue("BTC/USD", {})
    await engine.drain()
    with suppress(asyncio.CancelledError):
        await engine.stop()

    assert called == ["BTC/USD"]

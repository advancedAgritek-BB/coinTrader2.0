import asyncio
import logging
import types

import pytest

from crypto_bot import evaluator


@pytest.mark.asyncio
async def test_evaluate_batch_handles_outcomes(caplog, monkeypatch):
    async def eval_fn(symbol: str):
        if symbol == "good":
            return {"symbol": symbol}
        if symbol == "bad":
            raise RuntimeError("boom")
        await asyncio.sleep(0.02)

    ctx = types.SimpleNamespace(eval_fn=eval_fn)

    monkeypatch.setattr(evaluator, "DEFAULT_TIMEOUT", 0.01)

    with caplog.at_level(logging.INFO):
        results = await evaluator.evaluate_batch(["good", "bad", "slow"], ctx)

    assert results == {"good": {"symbol": "good"}}
    timeout_msg = next(r for r in caplog.records if "[EVAL TIMEOUT] slow" in r.message)
    error_msg = next(r for r in caplog.records if "[EVAL ERROR] bad" in r.message)
    assert timeout_msg
    assert error_msg

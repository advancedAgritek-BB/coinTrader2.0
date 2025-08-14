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

    class FakeMonotonic:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            val = self.t
            self.t += 1.0
            return val

    monkeypatch.setattr(evaluator, "DEFAULT_TIMEOUT", 0.01)
    monkeypatch.setattr(evaluator, "SUMMARY_INTERVAL", 2.5)
    monkeypatch.setattr(evaluator.time, "monotonic", FakeMonotonic())

    with caplog.at_level(logging.INFO):
        results = await evaluator.evaluate_batch(["good", "bad", "slow"], ctx)

    assert results == {"good": {"symbol": "good"}}
    timeout_msg = next(r for r in caplog.records if "[EVAL TIMEOUT] slow" in r.message)
    error_msg = next(r for r in caplog.records if "[EVAL ERROR] bad" in r.message)
    summary = next(r for r in caplog.records if "Eval stats" in r.message)
    assert timeout_msg
    assert error_msg
    assert "scanned=3" in summary.message
    assert "ok=1" in summary.message
    assert "errors=1" in summary.message
    assert "timeouts=1" in summary.message
    assert "signals=1" in summary.message

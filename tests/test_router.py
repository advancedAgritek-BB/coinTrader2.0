import asyncio
import pandas as pd
import logging

import crypto_bot.strategy_router as router
from crypto_bot.signals.signal_scoring import evaluate_async


def slow_strategy(df, cfg=None):
    import time
    time.sleep(6)
    return 1.0, "long"


def test_strategy_timeout_logged(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(router, "get_strategy_by_name", lambda n: slow_strategy)

    cfg = {"strategy_router": {"regimes": {"trending": ["slow"]}}}
    fn = router.route("trending", "cex", cfg)

    async def run():
        df = pd.DataFrame({"close": [1, 2]})
        return await evaluate_async([fn], df, {})

    res = asyncio.run(run())
    assert res == [(0.0, "none", None)]
    assert any("TIMEOUT" in r.getMessage() for r in caplog.records)

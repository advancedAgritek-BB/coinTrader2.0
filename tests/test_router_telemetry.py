import pandas as pd
import asyncio

import crypto_bot.strategy_router as sr
from crypto_bot.utils.telemetry import telemetry, dump


def test_router_telemetry_counters(monkeypatch):
    telemetry.reset()

    def ok(df, cfg=None):
        return 0.4, "long"

    def timeout_fn(df, cfg=None):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(sr, "get_strategies_for_regime", lambda r, c=None: [ok, timeout_fn])

    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    cfg = sr.RouterConfig.from_dict({})
    try:
        sr.evaluate_regime("trending", df, cfg)
    except asyncio.TimeoutError:
        pass

    fn = sr.route("trending", "cex", cfg)
    fn(df, {"symbol": "AAA"})

    out = dump()
    assert "router.signals_checked" in out
    assert "router.signal_returned" in out
    assert "router.signal_timeout" in out
    assert "router.symbol_locked" in out

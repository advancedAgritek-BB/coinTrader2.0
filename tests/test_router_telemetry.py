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
    monkeypatch.setattr(sr.Selector, "select", lambda self, df, regime, mode, notifier: ok)

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


def test_unknown_regime_short_circuit(monkeypatch):
    telemetry.reset()
    messages: list[str] = []

    class DummyNotifier:
        def notify(self, msg: str) -> None:  # pragma: no cover - simple notifier
            messages.append(msg)

    cfg = sr.RouterConfig.from_dict({"symbol": "AAA"})
    fn = sr.route("unknown", "cex", cfg, notifier=DummyNotifier())
    score, direction = fn(pd.DataFrame(), {"symbol": "AAA"})

    assert score == 0.0
    assert direction == "none"
    out = dump()
    assert "router.unknown_regime" in out
    assert messages

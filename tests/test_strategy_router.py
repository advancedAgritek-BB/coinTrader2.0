import pytest

from crypto_bot import strategy_router
from crypto_bot.strategy_router import strategy_for, route
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    mean_bot,
    breakout_bot,
    sniper_bot,
    micro_scalp_bot,
    bounce_scalper,
)


def test_strategy_for_mapping():
    assert strategy_for("trending") is trend_bot.generate_signal
    assert strategy_for("sideways") is grid_bot.generate_signal
    assert strategy_for("mean-reverting") is mean_bot.generate_signal
    assert strategy_for("breakout") is breakout_bot.generate_signal
    assert strategy_for("volatile") is sniper_bot.generate_signal
    assert strategy_for("scalp") is micro_scalp_bot.generate_signal
    assert strategy_for("bounce") is bounce_scalper.generate_signal
    assert strategy_for("unknown") is grid_bot.generate_signal


def test_route_notifier(monkeypatch):
    msgs = []

    class DummyNotifier:
        def notify(self, text):
            msgs.append(text)

    def dummy_signal(df, cfg=None):
        return 0.5, "long"

    monkeypatch.setitem(strategy_router.STRATEGY_MAP, "trending", dummy_signal)

    fn = route("trending", "cex", {}, DummyNotifier())
    score, direction = fn(None, {"symbol": "AAA"})

    assert score == 0.5
    assert direction == "long"
    assert msgs == ["\U0001F4C8 Signal: AAA \u2192 LONG | Confidence: 0.50"]

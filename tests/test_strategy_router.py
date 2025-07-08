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


SAMPLE_CFG = {
    "strategy_router": {
        "regimes": {
            "trending": ["trend"],
            "sideways": ["grid"],
            "mean-reverting": ["mean_bot"],
            "breakout": ["breakout_bot"],
            "volatile": ["sniper_bot"],
            "scalp": ["micro_scalp"],
            "bounce": ["bounce_scalper"],
        }
    }
}


def test_strategy_for_mapping():
    assert strategy_for("trending", SAMPLE_CFG) is trend_bot.generate_signal
    assert strategy_for("sideways", SAMPLE_CFG) is grid_bot.generate_signal
    assert strategy_for("mean-reverting", SAMPLE_CFG) is mean_bot.generate_signal
    assert strategy_for("breakout", SAMPLE_CFG) is breakout_bot.generate_signal
    assert strategy_for("volatile", SAMPLE_CFG) is sniper_bot.generate_signal
    assert strategy_for("scalp", SAMPLE_CFG) is micro_scalp_bot.generate_signal
    assert strategy_for("bounce", SAMPLE_CFG) is bounce_scalper.generate_signal
    assert strategy_for("unknown", SAMPLE_CFG) is grid_bot.generate_signal


def test_route_notifier(monkeypatch):
    msgs = []

    class DummyNotifier:
        def notify(self, text):
            msgs.append(text)

    def dummy_signal(df, cfg=None):
        return 0.5, "long"

    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda n: dummy_signal if n == "dummy" else None,
    )

    cfg = {"strategy_router": {"regimes": {"trending": ["dummy"]}}}

    fn = route("trending", "cex", cfg, DummyNotifier())
    score, direction = fn(None, {"symbol": "AAA"})

    assert score == 0.5
    assert direction == "long"
    assert msgs == ["\U0001F4C8 Signal: AAA \u2192 LONG | Confidence: 0.50"]


def test_route_multi_tf_combo(monkeypatch):
    def dummy(df, cfg=None):
        return 0.1, "long"

    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda n: dummy if n == "dummy" else None,
    )

    cfg = {"timeframe": "1m", "strategy_router": {"regimes": {"breakout": ["dummy"]}}}

    fn = route({"1m": "breakout", "15m": "trending"}, "cex", cfg)
    score, direction = fn(None)
    assert (score, direction) == (0.1, "long")

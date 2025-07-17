import pytest
import asyncio

from crypto_bot import strategy_router
from crypto_bot.strategy_router import strategy_for, route, RouterConfig
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    mean_bot,
    breakout_bot,
    sniper_bot,
    sniper_solana,
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
    data = {"strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"], "commit_lock_intervals": 3}}
    cfg = RouterConfig.from_dict(data)
    assert strategy_for("trending", cfg) is trend_bot.generate_signal
    assert strategy_for("sideways", cfg) is grid_bot.generate_signal
    assert strategy_for("mean-reverting", cfg) is mean_bot.generate_signal
    assert strategy_for("breakout", cfg) is breakout_bot.generate_signal
    assert strategy_for("volatile", cfg) is sniper_bot.generate_signal
    assert strategy_for("scalp", cfg) is micro_scalp_bot.generate_signal
    assert strategy_for("bounce", cfg) is bounce_scalper.generate_signal
    assert strategy_for("unknown", cfg) is grid_bot.generate_signal


def test_strategy_for_solana_scalping():
    from crypto_bot.strategy import solana_scalping

    data = {
        "strategy_router": {"regimes": {"scalp": ["solana_scalping"]}}
    }
    cfg = RouterConfig.from_dict(data)
    assert strategy_for("scalp", cfg) is solana_scalping.generate_signal
    import crypto_bot.strategy_router as sr
    sr._build_mappings_cached.cache_clear()
    sr._CONFIG_REGISTRY.clear()


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

    cfg = RouterConfig.from_dict({"strategy_router": {"regimes": {"trending": ["dummy"]}}})

    fn = route("trending", "cex", cfg, DummyNotifier())
    score, direction = asyncio.run(fn(None, {"symbol": "AAA"}))

    assert score == 0.5
    assert direction == "long"
    assert msgs == ["\U0001F4C8 Signal: AAA \u2192 LONG | Confidence: 0.50"]


def test_route_multi_tf_combo(monkeypatch, tmp_path):
    def dummy(df, cfg=None):
        return 0.1, "long"

    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda n: dummy if n == "dummy" else None,
    )

    monkeypatch.setattr(strategy_router.commit_lock, "LOG_DIR", tmp_path)
    monkeypatch.setattr(strategy_router, "LAST_REGIME_FILE", tmp_path / "last.json")

    cfg = RouterConfig.from_dict({"timeframe": "1m", "strategy_router": {"regimes": {"breakout": ["dummy"]}}})

    fn = route({"1m": "breakout", "15m": "trending"}, "cex", cfg)
    score, direction = asyncio.run(fn(pd.DataFrame()))
    assert (score, direction) == (0.1, "long")


def test_regime_commit_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_router.commit_lock, "LOG_DIR", tmp_path)

    data = {
        "strategy_router": {
            "regimes": SAMPLE_CFG["strategy_router"]["regimes"],
            "commit_lock_intervals": 3,
        }
    }
    cfg = RouterConfig.from_dict(data)
    route("trending", "cex", cfg)
    lock = tmp_path / "last_regime.json"
    ts = lock.stat().st_mtime

    fn = route("sideways", "cex", cfg)

    assert fn.__name__ == trend_bot.generate_signal.__name__
    assert lock.stat().st_mtime == ts

import pandas as pd
from crypto_bot.strategy_router import route
from crypto_bot.strategy import breakout_bot, trend_bot


def make_df(close_vals, vol_vals):
    idx = pd.date_range("2025-01-01", periods=len(close_vals), freq="T")
    return pd.DataFrame({"open": close_vals, "high": close_vals, "low": close_vals,
                         "close": close_vals, "volume": vol_vals}, index=idx)


def test_fastpath_breakout(tmp_path, monkeypatch):
    cfg = {"strategy_router": {"fast_path": {
        "breakout_squeeze_window": 5,
        "breakout_bandwidth_zscore": -0.84,
        "breakout_volume_multiplier": 2,
        "trend_adx_threshold": 1000
    }}, "regime": {"sideways": ["grid"]}}

    close = list(range(10))
    volume = [1] * 9 + [10]
    df = make_df(close, volume)
    fn = route("sideways", "cex", cfg, None, df)
    assert fn.__name__ == breakout_bot.generate_signal.__name__


def test_fastpath_trend(tmp_path):
    cfg = {"strategy_router": {"fast_path": {
        "breakout_squeeze_window": 3,
        "breakout_max_bandwidth": 0,
        "breakout_volume_multiplier": 100,
        "trend_adx_threshold": 5
    }}, "regime": {"trending": ["trend"]}}
    # create rising series so ADX > threshold
    vals = list(range(10))
    df = make_df(vals, [1]*10)
    fn = route("trending", "cex", cfg, None, df)
    assert fn.__name__ == trend_bot.generate_signal.__name__


def test_onchain_solana_route():
    cfg = {
        "chain": "sol",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "onchain", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__
    fn = route("volatile", "onchain", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__


def test_usdc_pair_breakout():
    cfg = {
        "symbol": "XYZ/USDC",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "cex", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__


def test_auto_solana_breakout_route():
    cfg = {
        "preferred_chain": "solana",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "auto", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__


def test_dynamic_grid_uses_micro_scalp():
    cfg = {
        "symbol": "AAA/USDT",
        "grid_bot": {"dynamic_grid": True},
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("sideways", "cex", cfg)
    assert fn.__name__ == micro_scalp_bot.generate_signal.__name__


def test_strategy_timeframe_routing(monkeypatch):
    captured = {}

    def dummy(df, cfg=None):
        captured['df'] = df
        return 0.2, "long"

    monkeypatch.setattr(strategy_router, "get_strategy_by_name", lambda n: dummy if n == "dummy" else None)

    data = {
        "timeframe": "1m",
        "breakout_timeframe": "5m",
        "strategy_router": {"regimes": {"breakout": ["dummy"]}},
    }
    cfg = RouterConfig.from_dict(data)

    df_map = {"1m": pd.DataFrame({"v": [1]}), "5m": pd.DataFrame({"v": [5]})}
    fn = route("breakout", "cex", cfg)
    asyncio.run(fn(df_map))
    assert captured.get('df') is df_map["5m"]


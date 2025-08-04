import pytest
import asyncio
import sys
import types
import json

import fakeredis

sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
if not hasattr(sys.modules["solana.rpc.async_api"], "AsyncClient"):
    class DummyClient:
        pass

    sys.modules["solana.rpc.async_api"].AsyncClient = DummyClient
sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
sys.modules.setdefault("solders.pubkey", types.ModuleType("solders.pubkey"))
sys.modules.setdefault("solders.rpc.responses", types.ModuleType("solders.rpc.responses"))
if not hasattr(sys.modules["solders.pubkey"], "Pubkey"):
    class DummyPubkey:
        @staticmethod
        def from_string(s):
            return s

    sys.modules["solders.pubkey"].Pubkey = DummyPubkey
if not hasattr(sys.modules["solders.rpc.responses"], "GetTokenAccountBalanceResp"):
    class DummyResp:
        pass

    sys.modules["solders.rpc.responses"].GetTokenAccountBalanceResp = DummyResp
    sys.modules["solders.rpc.responses"].GetAccountInfoResp = DummyResp

from crypto_bot import strategy_router
from crypto_bot.strategy_router import strategy_for, route, RouterConfig
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    mean_bot,
    dip_hunter,
    breakout_bot,
    sniper_bot,
    sniper_solana,
    micro_scalp_bot,
    bounce_scalper,
    flash_crash_bot,
)


SAMPLE_CFG = {
    "strategy_router": {
        "regimes": {
            "trending": ["trend", "momentum_bot"],
            "sideways": ["grid"],
            "mean-reverting": ["dip_hunter", "stat_arb_bot"],
            "breakout": ["breakout_bot"],
            "volatile": ["sniper_bot", "momentum_bot"],
            "scalp": ["micro_scalp"],
            "bounce": ["bounce_scalper"],
        }
    }
}


def test_strategy_for_mapping():
    data = {"strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"], "commit_lock_intervals": 3}}
    cfg = RouterConfig.from_dict(data)
    assert (
        strategy_for("trending", cfg).__name__
        == trend_bot.generate_signal.__name__
    )
    assert strategy_for("sideways", cfg).__name__ == grid_bot.generate_signal.__name__
    assert (
        strategy_for("mean-reverting", cfg).__name__
        == dip_hunter.generate_signal.__name__
    )
    assert (
        strategy_for("dip_hunter", cfg).__name__
        == dip_hunter.generate_signal.__name__
    )
    assert strategy_for("breakout", cfg).__name__ == breakout_bot.generate_signal.__name__
    assert strategy_for("volatile", cfg).__name__ == sniper_bot.generate_signal.__name__
    assert strategy_for("scalp", cfg).__name__ == micro_scalp_bot.generate_signal.__name__
    assert strategy_for("bounce", cfg).__name__ == bounce_scalper.generate_signal.__name__
    assert strategy_for("unknown", cfg).__name__ == sniper_bot.generate_signal.__name__


def test_strategy_for_momentum_bot():
    from crypto_bot.strategy import momentum_bot

    data = {"strategy_router": {"regimes": {"trending": ["momentum_bot"], "volatile": ["momentum_bot"]}}}
    cfg = RouterConfig.from_dict(data)

    assert strategy_for("trending", cfg).__name__ == momentum_bot.generate_signal.__name__
    assert strategy_for("volatile", cfg).__name__ == momentum_bot.generate_signal.__name__


def test_strategy_for_solana_scalping():
    from crypto_bot.strategy import solana_scalping

    data = {
        "strategy_router": {"regimes": {"scalp": ["solana_scalping"]}}
    }
    cfg = RouterConfig.from_dict(data)
    assert (
        strategy_for("scalp", cfg).__name__
        == solana_scalping.generate_signal.__name__
    )
    import crypto_bot.strategy_router as sr
    sr._build_mappings_cached.cache_clear()
    sr._CONFIG_REGISTRY.clear()


def test_strategy_for_cross_chain_arb_bot():
    from crypto_bot.strategy import cross_chain_arb_bot

    data = {
        "strategy_router": {"regimes": {"trending": ["cross_chain_arb_bot"]}}
    }
    cfg = RouterConfig.from_dict(data)
    assert (
        strategy_for("trending", cfg).__name__
        == cross_chain_arb_bot.generate_signal.__name__
    )


def test_route_returns_meme_wave_bot():
    from crypto_bot.strategy import meme_wave_bot

    cfg = {
        "strategy_router": {"regimes": {"volatile": ["meme_wave_bot"]}}
    }
    fn = route("volatile", "cex", cfg)
    assert fn.__name__ == meme_wave_bot.generate_signal.__name__


def test_route_returns_lstm_bot():
    from crypto_bot.strategy import lstm_bot

    cfg = {"strategy_router": {"regimes": {"trending": ["lstm_bot"]}}}
    fn = route("trending", "cex", cfg)
    assert fn.__name__ == lstm_bot.generate_signal.__name__


def test_route_handles_none_df_map():
    cfg = {"strategy_router": {"regimes": {"trending": ["trend"]}}}
    fn = route("trending", "cex", cfg, df_map=None)
    assert fn.__name__ == trend_bot.generate_signal.__name__
    score, direction = fn(None)
    assert isinstance(score, float)
    assert isinstance(direction, str)


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

    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(strategy_router.commit_lock, "REDIS_CLIENT", fake_r)
    monkeypatch.setattr(strategy_router, "LAST_REGIME_FILE", tmp_path / "last.json")

    cfg = RouterConfig.from_dict({"timeframe": "1m", "strategy_router": {"regimes": {"breakout": ["dummy"]}}})

    fn = route({"1m": "breakout", "15m": "trending"}, "cex", cfg)
    score, direction = fn(pd.DataFrame())
    assert (score, direction) == (0.1, "long")


def test_regime_commit_lock(monkeypatch):
    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(strategy_router.commit_lock, "REDIS_CLIENT", fake_r)
    import fakeredis
    import threading

    class FakeRedisWithLock(fakeredis.FakeRedis):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._lock = threading.Lock()

        def lock(self, name, blocking_timeout=None):
            return self._lock

    fake = FakeRedisWithLock(decode_responses=True)
    monkeypatch.setattr(
        strategy_router.commit_lock.redis,
        "Redis",
        lambda *a, **k: fake,
    )

    data = {
        "strategy_router": {
            "regimes": SAMPLE_CFG["strategy_router"]["regimes"],
            "commit_lock_intervals": 3,
        }
    }
    cfg = RouterConfig.from_dict(data)
    route("trending", "cex", cfg)
    stored = json.loads(fake_r.get(strategy_router.commit_lock.REDIS_KEY))
    ts = stored["timestamp"]
    key = "commit_lock:last_regime"
    first = fake.get(key)

    fn = route("sideways", "cex", cfg)

    assert fn.__name__ == trend_bot.generate_signal.__name__
    new = json.loads(fake_r.get(strategy_router.commit_lock.REDIS_KEY))
    assert new == stored
    assert fake.get(key) == first

import pandas as pd
from crypto_bot.strategy_router import route
from crypto_bot.strategy import breakout_bot, trend_bot, sniper_bot


def make_df(close_vals, vol_vals):
    idx = pd.date_range("2025-01-01", periods=len(close_vals), freq="T")
    return pd.DataFrame({"open": close_vals, "high": close_vals, "low": close_vals,
                         "close": close_vals, "volume": vol_vals}, index=idx)


def make_breakdown_df() -> pd.DataFrame:
    rows = 30
    close = [1 + i / (rows - 1) for i in range(rows)]
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    volume = [100 + i for i in range(rows)]
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2025-01-01", periods=rows, freq="T"),
    )
    df.loc[df.index[-1], "close"] = min(low) - 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = sum(volume) / len(volume) * 3
    return df


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


def test_fastpath_breakdown_sniper():
    cfg = {"strategy_router": {"fast_path": {"breakdown_window": 5, "breakdown_volume_multiplier": 2}}}
    df = make_breakdown_df()
    fn = route("sideways", "cex", cfg, None, df)
    assert fn.__name__ == sniper_bot.generate_signal.__name__


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
        "onchain_default_quote": "USDC",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "cex", cfg)
    # unsupported token should fall back to breakout_bot
    assert fn.__name__ == breakout_bot.generate_signal.__name__


def test_auto_solana_breakout_route():
    cfg = {
        "preferred_chain": "solana",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "auto", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__


def test_auto_usdc_pair_route(monkeypatch):
    monkeypatch.setitem(strategy_router.TOKEN_MINTS, "XYZ", "mint")
    cfg = {
        "symbol": "XYZ/USDC",
        "onchain_default_quote": "USDC",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "auto", cfg)
    assert fn.__name__ == sniper_solana.generate_signal.__name__


def test_auto_usdc_pair_cex_fallback(monkeypatch):
    monkeypatch.setattr(strategy_router, "TOKEN_MINTS", {})
    cfg = {
        "symbol": "XYZ/USDC",
        "onchain_default_quote": "USDC",
        "strategy_router": {"regimes": SAMPLE_CFG["strategy_router"]["regimes"]},
    }
    fn = route("breakout", "auto", cfg)
    assert fn.__name__ == breakout_bot.generate_signal.__name__


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
    fn(df_map)
    assert captured.get('df') is df_map["5m"]


def test_flash_crash_timeframe_override(monkeypatch):
    captured = {}

    def dummy(df, cfg=None):
        captured['df'] = df
        return 0.0, "none"

    monkeypatch.setattr(flash_crash_bot, "generate_signal", dummy)
    import crypto_bot.meta_selector as meta_selector

    monkeypatch.setitem(
        meta_selector._STRATEGY_FN_MAP,
        "flash_crash_bot",
        dummy,
    )

    data = {
        "mean_reverting_timeframe": "1h",
        "flash_crash_timeframe": "1m",
        "strategy_router": {"regimes": {"mean-reverting": ["flash_crash_bot"]}},
    }
    cfg = RouterConfig.from_dict(data)

    df_map = {"1m": pd.DataFrame({"v": [1]}), "1h": pd.DataFrame({"v": [60]})}
    fn = route("mean-reverting", "cex", cfg)
    fn(df_map)
    assert captured.get("df") is df_map["1m"]


def test_route_mempool_blocks_signal(monkeypatch):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": volumes})

    class DummyMonitor:
        def is_suspicious(self, threshold):
            return True

    cfg = {
        "strategy_router": {"regimes": {"scalp": ["micro_scalp"]}},
        "micro_scalp": {"fresh_cross_only": False, "min_vol_z": 0},
        "mempool_monitor": {"enabled": True, "suspicious_fee_threshold": 1},
    }
    fn = route(
        "scalp",
        "cex",
        cfg,
        mempool_monitor=DummyMonitor(),
        mempool_cfg=cfg["mempool_monitor"],
    )
    score, direction = fn(df, cfg)
    assert (score, direction) == (0.0, "none")


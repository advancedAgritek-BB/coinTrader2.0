import pandas as pd
import numpy as np
import asyncio

from crypto_bot.regime.regime_classifier import (
    classify_regime,
    classify_regime_async,
)
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy_router import strategy_for
import crypto_bot.strategy_router as strategy_router
from crypto_bot.signals.signal_scoring import evaluate_async
import crypto_bot.signals.signal_scoring as sc


def test_classify_regime_returns_unknown_for_short_df():
    data = {
        "open": list(range(10)),
        "high": list(range(1, 11)),
        "low": list(range(10)),
        "close": list(range(10)),
        "volume": [100] * 10,
    }
    df = pd.DataFrame(data)
    regime, patterns = classify_regime(df)
    assert regime == "unknown"


def test_classify_regime_returns_unknown_for_14_rows():
    data = {
        "open": list(range(14)),
        "high": list(range(1, 15)),
        "low": list(range(14)),
        "close": list(range(14)),
        "volume": [100] * 14,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df)[0] == "unknown"


def test_classify_regime_returns_unknown_between_15_and_19_rows():
    for rows in range(15, 20):
        data = {
            "open": list(range(rows)),
            "high": list(range(1, rows + 1)),
            "low": list(range(rows)),
            "close": list(range(rows)),
            "volume": [100] * rows,
        }
        df = pd.DataFrame(data)
        assert classify_regime(df)[0] == "unknown"
def test_classify_regime_handles_index_error(monkeypatch):
    data = {
        "open": list(range(30)),
        "high": list(range(1, 31)),
        "low": list(range(30)),
        "close": list(range(30)),
        "volume": [100] * 30,
    }
    df = pd.DataFrame(data)

    def raise_index(*args, **kwargs):
        raise IndexError

    monkeypatch.setattr(
        __import__("ta").trend, "adx", raise_index
    )

    assert classify_regime(df)[0] == "unknown"


def test_classify_regime_uses_custom_thresholds(tmp_path):
    rows = 50
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    volume = np.arange(rows) + 100
    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # With default config this should be trending
    assert classify_regime(df)[0] == "trending"

    custom_cfg = tmp_path / "regime.yaml"
    custom_cfg.write_text(
        """\
adx_trending_min: 101
adx_sideways_max: 20
bb_width_sideways_max: 5
bb_width_breakout_max: 4
breakout_volume_mult: 2
rsi_mean_rev_min: 30
rsi_mean_rev_max: 70
ema_distance_mean_rev_max: 0.01
atr_volatility_mult: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
"""
    )

    # ADX threshold is too high so regime should no longer be trending
    assert classify_regime(df, config_path=str(custom_cfg))[0] != "trending"


def _make_trending_df(rows: int = 50) -> pd.DataFrame:
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    volume = np.arange(rows) + 100
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_trend_confirmed_by_higher_timeframe(tmp_path):
    df_low = _make_trending_df()
    df_high = _make_trending_df()

    cfg = tmp_path / "regime.yaml"
    cfg.write_text(
        """\
adx_trending_min: 25
adx_sideways_max: 20
bb_width_sideways_max: 5
bb_width_breakout_max: 4
breakout_volume_mult: 2
rsi_mean_rev_min: 30
rsi_mean_rev_max: 70
ema_distance_mean_rev_max: 0.01
atr_volatility_mult: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
higher_timeframe: '4h'
confirm_trend_with_higher_tf: true
"""
    )

    regime, _ = classify_regime(df_low, df_high, config_path=str(cfg))
    assert regime == "trending"


def test_trend_not_confirmed_when_higher_timeframe_disagrees(tmp_path):
    df_low = _make_trending_df()
    rows = 50
    const = np.ones(rows)
    df_high = pd.DataFrame({
        "open": const,
        "high": const + 0.1,
        "low": const - 0.1,
        "close": const,
        "volume": np.arange(rows) + 100,
    })

    cfg = tmp_path / "regime.yaml"
    cfg.write_text(
        """\
adx_trending_min: 25
adx_sideways_max: 20
bb_width_sideways_max: 5
bb_width_breakout_max: 4
breakout_volume_mult: 2
rsi_mean_rev_min: 30
rsi_mean_rev_max: 70
ema_distance_mean_rev_max: 0.01
atr_volatility_mult: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
higher_timeframe: '4h'
confirm_trend_with_higher_tf: true
"""
    )

    regime, _ = classify_regime(df_low, df_high, config_path=str(cfg))
    assert regime != "trending"


def test_classify_regime_async_matches_sync():
    df = _make_trending_df()
    sync_res = classify_regime(df)

    async def run():
        return await classify_regime_async(df)

    async_res = asyncio.run(run())
    assert async_res == sync_res


def test_analyze_symbol_async_consistent():
    df = _make_trending_df()
    regime, _ = classify_regime(df)
    strategy = strategy_for(regime)
    sync_score, sync_dir = asyncio.run(evaluate_async(strategy, df, {}))

    async def run():
        cfg = {"timeframe": "1h", "regime_timeframes": ["5m", "15m", "1h"], "min_consistent_agreement": 2}
        df_map = {"5m": df, "15m": df, "1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["regime"] == regime
    assert isinstance(res.get("patterns"), set)
    assert res["confidence"] == 1.0
    assert res["score"] == sync_score
    assert res["direction"] == sync_dir


def test_analyze_symbol_best_mode(monkeypatch):
    df = _make_trending_df()

    def strat_a(df, cfg=None):
        return 0.2, "long"

    def strat_b(df, cfg=None):
        return 0.7, "short"

    monkeypatch.setattr(strategy_router, "get_strategies_for_regime", lambda r: [strat_a, strat_b])
    eval_stub = lambda strats, df_, cfg_: {"score": 0.7, "direction": "short", "name": "strat_b"}
    monkeypatch.setattr(sc, "evaluate_strategies", eval_stub)
    monkeypatch.setattr(strategy_router, "evaluate_strategies", eval_stub, raising=False)
    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "evaluate_strategies", eval_stub)

    async def run():
        cfg = {
            "timeframe": "1h",
            "strategy_evaluation_mode": "best",
            "scoring_weights": {"strategy_score": 1.0},
            "regime_timeframes": ["5m", "15m", "1h"],
            "min_consistent_agreement": 2,
        }
        df_map = {"5m": df, "15m": df, "1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["name"] == "strat_b"
    assert res["direction"] == "short"
    assert res["score"] == 0.7


def test_voting_direction_override(monkeypatch):
    df = _make_trending_df()

    def base(df, cfg=None):
        return 0.4, "short"

    def v1(df, cfg=None):
        return 0.2, "long"

    def v2(df, cfg=None):
        return 0.3, "long"

    def v3(df, cfg=None):
        return 0.1, "short"

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "route", lambda *a, **k: base)
    monkeypatch.setattr(strategy_router, "route", lambda *a, **k: base)
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"a": v1, "b": v2, "c": v3}.get(n))
    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda name: {"a": v1, "b": v2, "c": v3}.get(name),
    )

    async def run():
        cfg = {
            "timeframe": "1h",
            "regime_timeframes": ["1h"],
            "voting_strategies": ["a", "b", "c"],
            "min_agreeing_votes": 2,
        }
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["direction"] == "long"


def test_voting_no_consensus(monkeypatch):
    df = _make_trending_df()

    def base(df, cfg=None):
        return 0.6, "long"

    def v1(df, cfg=None):
        return 0.2, "long"

    def v2(df, cfg=None):
        return 0.3, "short"

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "route", lambda *a, **k: base)
    monkeypatch.setattr(strategy_router, "route", lambda *a, **k: base)
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"a": v1, "b": v2}.get(n))
    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda name: {"a": v1, "b": v2}.get(name),
    )

    async def run():
        cfg = {
            "timeframe": "1h",
            "regime_timeframes": ["1h"],
            "voting_strategies": ["a", "b"],
            "min_agreeing_votes": 2,
        }
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["direction"] == "none"

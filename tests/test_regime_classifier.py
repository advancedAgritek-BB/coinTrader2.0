import pandas as pd
import numpy as np
import asyncio
import pytest
import yaml
import sys
import types
import os

from crypto_bot.regime.regime_classifier import (
    classify_regime,
    classify_regime_async,
    classify_regime_with_patterns,
)
from crypto_bot.regime import regime_classifier as rc
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy_router import strategy_for
import crypto_bot.strategy_router as strategy_router
from crypto_bot.signals.signal_scoring import evaluate_async
import crypto_bot.signals.signal_scoring as sc
from crypto_bot.utils.telemetry import telemetry


class DummyModel:
    def predict(self, X):
        return [0.8]


@pytest.fixture(autouse=True)
def reset_telemetry():
    telemetry.reset()
    yield


def test_classify_regime_returns_trending_for_short_df():
    data = {
        "open": list(range(10)),
        "high": list(range(1, 11)),
        "low": list(range(10)),
        "close": list(range(10)),
        "volume": [100] * 10,
    }
    df = pd.DataFrame(data)
    regime, probs = classify_regime(df)
    assert regime == "trending"
    assert isinstance(probs, dict)

    regime, info = classify_regime(df)
    assert regime == "trending"
    assert isinstance(info, dict)


def test_classify_regime_returns_trending_for_14_rows():
    data = {
        "open": list(range(14)),
        "high": list(range(1, 15)),
        "low": list(range(14)),
        "close": list(range(14)),
        "volume": [100] * 14,
    }
    df = pd.DataFrame(data)
    label, conf = classify_regime(df)
    assert label == "trending"
    assert isinstance(conf, dict)
    label, _ = classify_regime(df)
    assert isinstance(label, str)


def test_classify_regime_handles_none_df():
    label, probs = classify_regime(None)
    assert label == "unknown"
    assert probs == {"unknown": 0.0}


def test_classify_regime_returns_trending_between_15_and_19_rows():
    for rows in range(15, 20):
        data = {
            "open": list(range(rows)),
            "high": list(range(1, rows + 1)),
            "low": list(range(rows)),
            "close": list(range(rows)),
            "volume": [100] * rows,
        }
        df = pd.DataFrame(data)
        assert classify_regime(df)[0] == "trending"


def test_adx_trending_min_threshold(tmp_path):
    df = _make_trending_df()

    low_cfg = rc.CONFIG.copy()
    low_cfg["adx_trending_min"] = 10
    low_path = tmp_path / "low.yaml"
    low_path.write_text(yaml.safe_dump(low_cfg))
    assert classify_regime(df, config_path=str(low_path))[0] == "trending"

    high_cfg = rc.CONFIG.copy()
    high_cfg["adx_trending_min"] = 110
    high_cfg["adx_sideways_max"] = 200
    high_cfg["bb_width_sideways_max"] = 50
    high_path = tmp_path / "high.yaml"
    high_path.write_text(yaml.safe_dump(high_cfg))
    assert classify_regime(df, config_path=str(high_path))[0] == "sideways"
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

    assert classify_regime(df)[0] == "trending"
    label, conf = classify_regime(df)
    assert label == "trending"
    assert isinstance(conf, dict)
    assert isinstance(classify_regime(df)[0], str)



def test_hft_thresholds_override(tmp_path, monkeypatch):
    df = _make_trending_df()
    cfg = rc.CONFIG.copy()
    cfg["indicator_window"] = 14
    cfg["adx_trending_min"] = 15
    cfg["hft_indicator_window"] = 7
    cfg["hft_adx_trending_min"] = 30
    cfg["atr_baseline"] = None
    path = tmp_path / "hft.yaml"
    path.write_text(yaml.safe_dump(cfg))

    captured: dict[str, int] = {}

    def fake_classify_all(df, higher_df, cfg_local, *, df_map=None):
        captured["indicator_window"] = cfg_local.get("indicator_window")
        captured["adx_trending_min"] = cfg_local.get("adx_trending_min")
        return "trending", {}, {}

    monkeypatch.setattr(rc, "_classify_all", fake_classify_all)

    classify_regime(df, config_path=str(path), timeframe="30s")

    assert captured["indicator_window"] == cfg["hft_indicator_window"]
    assert captured["adx_trending_min"] == cfg["hft_adx_trending_min"]



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


def _make_sideways_df(rows: int = 50) -> pd.DataFrame:
    const = np.ones(rows)
    return pd.DataFrame({
        "open": const,
        "high": const + 0.1,
        "low": const - 0.1,
        "close": const,
        "volume": np.arange(rows) + 100,
    })


def _make_breakout_df(rows: int = 30) -> pd.DataFrame:
    df = _make_trending_df(rows)
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2
    return df


def _make_ascending_triangle_df(rows: int = 30) -> pd.DataFrame:
    df = _make_trending_df(rows)
    start = rows - 5
    high = df["high"].iloc[start]
    base_low = df["low"].iloc[start]
    for i in range(start, rows):
        df.loc[df.index[i], "high"] = high
        df.loc[df.index[i], "low"] = base_low + (i - start) * 0.02
        df.loc[df.index[i], "close"] = (df.loc[df.index[i], "high"] + df.loc[df.index[i], "low"]) / 2
    return df


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
normalized_range_volatility_min: 1.5
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
normalized_range_volatility_min: 1.5
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


def test_classify_regime_multi_timeframe_dict():
    df = _make_trending_df()
    res = classify_regime(df_map={"1h": df, "15m": df, "1d": df})
    assert isinstance(res, dict)
    assert set(res.values()) == {"trending"}


def test_classify_regime_two_timeframe_tuple():
    df = _make_trending_df()
    high = _make_trending_df()
    res = classify_regime(df_map={"1h": df, "4h": high})
    assert isinstance(res, tuple)
    assert res == ("trending", "trending")


def test_analyze_symbol_async_consistent():
    df = _make_trending_df()
    regime, _ = classify_regime(df)
    strategy = strategy_for(regime)
    sync_score, sync_dir, _ = asyncio.run(evaluate_async([strategy], df, {}))[0]

    async def run():
        cfg = {"timeframe": "1h", "regime_timeframes": ["5m", "15m", "1h"], "min_consistent_agreement": 2}
        df_map = {"5m": df, "15m": df, "1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["regime"] == regime
    assert isinstance(res.get("patterns"), dict)
    assert res["confidence"] == 1.0
    expected = sync_score / (1 + np.log1p(df["volume"].iloc[-1]) / 10)
    assert res["score"] == pytest.approx(expected)
    assert res["direction"] == sync_dir


def test_analyze_symbol_best_mode(monkeypatch):
    df = _make_trending_df()

    def strat_a(df, cfg=None):
        return 0.2, "long"

    def strat_b(df, cfg=None):
        return 0.7, "short"

    monkeypatch.setattr(strategy_router, "get_strategies_for_regime", lambda r, c=None: [strat_a, strat_b])
    eval_stub = lambda strats, df_, cfg_: {"score": 0.7, "direction": "short", "name": "strat_b"}
    monkeypatch.setattr(sc, "evaluate_strategies", eval_stub)
    monkeypatch.setattr(strategy_router, "evaluate_strategies", eval_stub, raising=False)
    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "evaluate_strategies", eval_stub)
    monkeypatch.setattr(ma, "get_strategies_for_regime", lambda r, c=None: [strat_a, strat_b])
    calls = []
    monkeypatch.setattr(ma, "log_second_place", lambda *a, **k: calls.append(a))

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
    expected = 0.7 / (1 + np.log1p(df["volume"].iloc[-1]) / 10)
    assert res["score"] == pytest.approx(expected)
    assert calls


def test_analyze_symbol_accepts_dict_patterns(monkeypatch):
    df = _make_trending_df()

    async def fake_async(*_a, **_k):
        return "trending", {"foo": 1.0}

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "classify_regime_async", fake_async)
    monkeypatch.setattr(ma, "classify_regime_cached", lambda *a, **k: ("trending", {"foo": 1.0}))

    async def run():
        cfg = {"timeframe": "1h"}
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["patterns"] == {"foo": 1.0}
    assert isinstance(res["patterns"], dict)


def test_analyze_symbol_handles_missing_df():
    df = _make_trending_df()

    async def run():
        cfg = {"timeframe": "1h"}
        df_map = {"5m": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res == {"symbol": "AAA", "skip": "no_ohlcv"}
    assert telemetry.snapshot().get("analysis.skipped_no_df", 0) == 1


def test_analyze_symbol_skips_short_data():
    df = _make_trending_df(10)

    async def run():
        cfg = {"timeframe": "1h"}
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res == {"symbol": "AAA", "skip": "short_data"}
    assert telemetry.snapshot().get("analysis.skipped_short_data", 0) == 1


def test_analyze_symbol_probabilities_match_regime():
    df = _make_trending_df()

    async def run():
        cfg = {"timeframe": "1h"}
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["regime"] in res["probabilities"]
    assert res["sub_regime"] in res["probabilities"]


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
            "voting_strategies": {
                "strategies": ["a", "b", "c"],
                "min_agreeing_votes": 2,
            },
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
            "voting_strategies": {
                "strategies": ["a", "b"],
                "min_agreeing_votes": 2,
            },
        }
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["direction"] == "none"


def test_voting_single_vote(monkeypatch):
    df = _make_trending_df()

    def base(df, cfg=None):
        return 0.6, "long"

    def v1(df, cfg=None):
        return 0.2, "long"

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "route", lambda *a, **k: base)
    monkeypatch.setattr(strategy_router, "route", lambda *a, **k: base)
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"a": v1}.get(n))
    monkeypatch.setattr(
        strategy_router,
        "get_strategy_by_name",
        lambda name: {"a": v1}.get(name),
    )

    async def run():
        cfg = {
            "timeframe": "1h",
            "regime_timeframes": ["1h"],
            "voting_strategies": {
                "strategies": ["a"],
                "min_agreeing_votes": 1,
            },
        }
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["direction"] == "long"


def test_regime_voting_disagreement_unknown():
    df_trend = _make_trending_df()
    df_side = _make_sideways_df()
    df_break = _make_breakout_df()

    async def run():
        cfg = {
            "timeframe": "1h",
            "regime_timeframes": ["5m", "15m", "1h"],
            "min_consistent_agreement": 2,
        }
        df_map = {"5m": df_trend, "15m": df_side, "1h": df_break}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["regime"] == "sideways"
    assert res["confidence"] == pytest.approx(2 / 3)


def test_breakout_pattern_sets_regime():
    df = _make_breakout_df()
    regime, patterns = classify_regime_with_patterns(df)
    regime, patterns = classify_regime(df)
    assert regime == "sideways"
    assert patterns.get("breakout", 0) >= 1.0
    assert regime == "breakout"
    assert isinstance(patterns, dict)
    assert patterns.get("breakout", 0) > 0
    assert "breakout" in patterns
    assert patterns["breakout"] > 0
    assert isinstance(patterns["breakout"], float)


def test_ascending_triangle_promotes_breakout():
    df = _make_ascending_triangle_df()
    regime, patterns = classify_regime(df)
    assert "ascending_triangle" in patterns
    assert regime == "breakout"


def test_volume_spike_triggers_breakout():
    df = _make_sideways_df()
    df.loc[df.index[-1], "volume"] = df.loc[df.index[-2], "volume"] * 2
    assert classify_regime(df)[0] == "breakout"


def test_high_volume_zscore_breakout():
    df = _make_sideways_df()
    df["volume"] = 100
    df.loc[df.index[-1], "volume"] = 200
    assert classify_regime(df)[0] == "breakout"


def test_ml_fallback_does_not_trigger_on_short_data(monkeypatch, tmp_path):
    df = _make_trending_df(5)
    called = False

    def fake(_df):
        nonlocal called
        called = True
        return "trending"

    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._ml_fallback",
        lambda _df: (fake(_df), 1.0),
    )

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
normalized_range_volatility_min: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
higher_timeframe: '4h'
confirm_trend_with_higher_tf: false
use_ml_regime_classifier: true
ml_min_bars: 5
"""
    )

    regime, _ = classify_regime(df, config_path=str(cfg))
    assert regime == "trending"
    assert not called


def test_ml_fallback_used_when_unknown(monkeypatch, tmp_path):
    df = _make_trending_df(30)
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._classify_core", lambda *_a, **_k: "unknown"
    )
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._ml_fallback", lambda _df: ("trending", 1.0)
    )
    assert classify_regime(df)[0] == "trending"
    label, conf = classify_regime(df)
    assert label == "trending"
    assert isinstance(conf, dict)
    assert isinstance(classify_regime(df)[0], str)

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
normalized_range_volatility_min: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
higher_timeframe: '4h'
confirm_trend_with_higher_tf: false
use_ml_regime_classifier: true
ml_min_bars: 5
"""
    )

    regime, patterns = classify_regime_with_patterns(df, config_path=str(cfg))
    regime, _ = classify_regime(df, config_path=str(cfg))
    assert regime == "trending"
    assert isinstance(patterns, dict)


def test_fallback_scores_when_indicator_unknown(monkeypatch, tmp_path):
    df = _make_trending_df(30)
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._classify_core", lambda *_a, **_k: "unknown"
    )
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._ml_fallback", lambda _df: ("unknown", 0.0)
    )

    cfg = tmp_path / "regime.yaml"
#    cfg.write_text(
#        """\
#adx_trending_min: 25
#adx_sideways_max: 20
#bb_width_sideways_max: 5
#bb_width_breakout_max: 4
#breakout_volume_mult: 2
#rsi_mean_rev_min: 30
#rsi_mean_rev_max: 70
#ema_distance_mean_rev_max: 0.01
#atr_volatility_mult: 1.5
#    assert patterns == {}
#    assert patterns == set()
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
"""
    )
    regime, patterns = classify_regime_with_patterns(df, config_path=str(cfg))
    regime, _ = classify_regime(df, config_path=str(cfg))
    assert regime == "trending"
    assert patterns == {}
    assert isinstance(patterns, dict)



def _make_volatility_df(base_range: float, last_range: float, rows: int = 50) -> pd.DataFrame:
    close = np.ones(rows)
    high = close + base_range / 2
    low = close - base_range / 2
    volume = [100] * rows
    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.loc[df.index[-1], "high"] = 1 + last_range / 2
    df.loc[df.index[-1], "low"] = 1 - last_range / 2
    return df


def test_normalized_range_volatile_detection(tmp_path):
    df = _make_volatility_df(0.1, 0.2)
    cfg = tmp_path / "regime.yaml"
    cfg.write_text(
        """\
adx_trending_min: 100
adx_sideways_max: 0
bb_width_sideways_max: 0
bb_width_breakout_max: 0
breakout_volume_mult: 100
rsi_mean_rev_min: 0
rsi_mean_rev_max: 0
ema_distance_mean_rev_max: 0
atr_volatility_mult: 1.5
normalized_range_volatility_min: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
higher_timeframe: '4h'
confirm_trend_with_higher_tf: false
use_ml_regime_classifier: true
ml_min_bars: 5
"""
    )

    label, confidence = classify_regime(df, config_path=str(cfg))
    assert label in {"trending", "mean-reverting", "sideways"}
    assert 0.0 <= confidence <= 1.0
    regime, _ = classify_regime(df, config_path=str(cfg))
    assert regime == "volatile"


def test_normalized_range_not_volatile_when_atr_high(tmp_path):
    df = _make_volatility_df(5, 6)
    cfg = tmp_path / "regime.yaml"
    cfg.write_text(
        """\
adx_trending_min: 100
adx_sideways_max: 0
bb_width_sideways_max: 0
bb_width_breakout_max: 0
breakout_volume_mult: 100
rsi_mean_rev_min: 0
rsi_mean_rev_max: 0
ema_distance_mean_rev_max: 0
atr_volatility_mult: 1.5
normalized_range_volatility_min: 1.5
ema_fast: 20
ema_slow: 50
indicator_window: 14
bb_window: 20
ma_window: 20
"""
    )
    regime, info = classify_regime(df, config_path=str(cfg))
    assert regime != "volatile"
    assert isinstance(info, dict)


def test_classify_regime_probabilities_sum_to_one():
    df = _make_trending_df()
    label, probs = classify_regime(df)
    assert label == "trending"
    assert pytest.approx(sum(probs.values())) == 1.0


def test_dip_hunter_regime_low_adx(tmp_path):
    df = _make_sideways_df()
    cfg = tmp_path / "regime.yaml"
    cfg.write_text("dip_hunter_adx_max: 25\n")
    label, _ = classify_regime(df, config_path=str(cfg))
    assert label == "dip_hunter"


def test_patterns_override_unknown(monkeypatch):
    df = _make_breakout_df()
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._classify_core", lambda *_a, **_k: "unknown"
    )
    regime, probs = classify_regime(df)
    assert regime == "breakout"
    assert probs["breakout"] == 1.0


def test_analyze_symbol_routes_to_dip_hunter():
    df = _make_sideways_df()

    async def run():
        cfg = {"timeframe": "1h"}
        df_map = {"1h": df}
        return await analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["regime"] == "dip_hunter"


def test_ml_blending(monkeypatch, tmp_path):
    df = _make_sideways_df(30)
    monkeypatch.setattr(
        "crypto_bot.regime.regime_classifier._classify_core", lambda *_a, **_k: "unknown"
    )
    monkeypatch.setattr(
        "crypto_bot.regime.ml_fallback.predict_regime",
        lambda _df: ("trending", 0.7),
    )
    cfg = tmp_path / "regime.yaml"
    cfg.write_text("use_ml_regime_classifier: true\nml_min_bars: 5\n")
    regime, probs = classify_regime(df, config_path=str(cfg))
    assert regime == "trending"
    assert probs["trending"] == pytest.approx(0.7)
    assert isinstance(patterns, set)


def test_adaptive_thresholds(tmp_path):
    df = _make_volatility_df(0.1, 0.1)
    cfg_file = tmp_path / "regime.yaml"
    cfg_file.write_text(
        """\
adx_trending_min: 25
adx_sideways_max: 18
bb_width_sideways_max: 0.025
bb_width_breakout_max: 4
breakout_volume_mult: 1.5
rsi_mean_rev_min: 30
rsi_mean_rev_max: 70
ema_distance_mean_rev_max: 0.02
atr_volatility_mult: 1.5
normalized_range_volatility_min: 1.5
atr_baseline: 0.01
ema_fast: 8
ema_slow: 21
indicator_window: 14
bb_window: 20
ma_window: 20
"""
    )
    cfg = rc._load_config(cfg_file)
    adapted = rc.adaptive_thresholds(cfg, df, "AAA")
    expected = 2 * cfg["adx_trending_min"]
    assert adapted["adx_trending_min"] == pytest.approx(expected)
    expected_vol = 2 * cfg["normalized_range_volatility_min"]
    assert adapted["normalized_range_volatility_min"] == pytest.approx(expected_vol)
    assert adapted["rsi_mean_rev_max"] >= cfg["rsi_mean_rev_max"]


def test_supabase_missing_env_uses_fallback(monkeypatch):
    df = _make_trending_df()
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    called = False

    def fallback(_df):
        nonlocal called
        called = True
        return "trending", 0.9

    monkeypatch.setattr(rc, "_ml_fallback", fallback)
    rc._supabase_model = None
    rc._supabase_scaler = None
    label, conf = rc._classify_ml(df)
    assert called
    assert label == "trending"
    assert conf == 0.9


def test_supabase_download_failure(monkeypatch):
    df = _make_trending_df()
    monkeypatch.setenv("SUPABASE_URL", "http://example.com")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    class FakeBucket:
        def download(self, path):
            raise Exception("fail")

    class FakeStorage:
        def from_(self, name):
            return FakeBucket()

    class FakeClient:
        def __init__(self, *a, **k):
            self.storage = FakeStorage()

    monkeypatch.setattr(rc, "_supabase_model", None)
    monkeypatch.setattr(rc, "_supabase_scaler", None)
    monkeypatch.setattr(rc, "_ml_fallback", lambda _df: ("sideways", 0.5))
    monkeypatch.setitem(sys.modules, "supabase", types.SimpleNamespace(create_client=lambda u, k: FakeClient()))

    label, conf = rc._classify_ml(df)
    assert label == "sideways"
    assert conf == 0.5


def test_supabase_latest_missing_loads_direct_model(monkeypatch):
    import pickle

    df = _make_trending_df()
    monkeypatch.setenv("SUPABASE_URL", "http://example.com")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    model_bytes = pickle.dumps(DummyModel())

    class NotFound(Exception):
        def __init__(self):
            self.message = "not_found"

    class FakeBucket:
        def download(self, path):
            if path.endswith("LATEST.json"):
                raise NotFound()
            assert path.endswith("xrpusd_regime_lgbm.pkl")
            return model_bytes

    class FakeStorage:
        def from_(self, name):
            return FakeBucket()

    class FakeClient:
        def __init__(self, *a, **k):
            self.storage = FakeStorage()

    called = False

    def fallback(_df, _notifier=None):
        nonlocal called
        called = True
        return "sideways", 0.5

    monkeypatch.setattr(rc, "_ml_fallback", fallback)
    monkeypatch.setattr(rc, "_supabase_model", None)
    monkeypatch.setattr(rc, "_supabase_scaler", None)
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "supabase",
        types.SimpleNamespace(create_client=lambda u, k: FakeClient()),
    )

    label, conf = rc._classify_ml(df)
    assert label == "trending"
    assert conf > 0
    assert not called


def test_hft_env_overrides(monkeypatch):
    df = _make_trending_df()

    captured = {}

    def fake_classify_all(df, higher_df, cfg, *, df_map=None):
        captured.update(cfg)
        return "trending", {}, {}

    monkeypatch.setattr(rc, "_classify_all", fake_classify_all)

    envs = {
        "HFT_ADX_MIN": "99",
        "HFT_RSI_MIN": "1",
        "HFT_RSI_MAX": "2",
        "HFT_NR_VOL_MIN": "3",
        "HFT_INDICATOR_WINDOW": "4",
        "HFT_ML_BLEND_WEIGHT": "0.5",
    }
    for k, v in envs.items():
        monkeypatch.setenv(k, v)

    classify_regime(df, timeframe="30s")

    assert captured["adx_trending_min"] == 99.0
    assert captured["rsi_mean_rev_min"] == 1.0
    assert captured["rsi_mean_rev_max"] == 2.0
    assert captured["normalized_range_volatility_min"] == 3.0
    assert captured["indicator_window"] == 4
    assert captured["ml_blend_weight"] == 0.5


@pytest.mark.asyncio
async def test_analyze_symbol_flags_too_flat(monkeypatch):
    import crypto_bot.utils.market_analyzer as ma

    async def fake_async(*_a, **_k):
        return "unknown", {"unknown": 1.0}

    monkeypatch.setattr(ma, "ML_AVAILABLE", True)
    monkeypatch.setattr(ma, "classify_regime_async", fake_async)
    monkeypatch.setattr(ma, "detect_patterns", lambda _df: {})

    price = np.ones(60)
    df = pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.ones(60),
        }
    )
    cfg = {
        "timeframe": "1h",
        "regime_timeframes": [],
        "volatility_filter": {"min_atr_pct": 0.1},
    }
    res = await ma.analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)
    assert res["too_flat"] is True



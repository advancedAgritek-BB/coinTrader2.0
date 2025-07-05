import pandas as pd
import numpy as np
import asyncio

from crypto_bot.regime.regime_classifier import classify_regime, classify_regime_async
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy_router import strategy_for
from crypto_bot.signals.signal_scoring import evaluate_async


def test_classify_regime_returns_unknown_for_short_df():
    data = {
        "open": list(range(10)),
        "high": list(range(1, 11)),
        "low": list(range(10)),
        "close": list(range(10)),
        "volume": [100] * 10,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df) == "unknown"


def test_classify_regime_returns_unknown_for_14_rows():
    data = {
        "open": list(range(14)),
        "high": list(range(1, 15)),
        "low": list(range(14)),
        "close": list(range(14)),
        "volume": [100] * 14,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df) == "unknown"


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
        assert classify_regime(df) == "unknown"
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

    assert classify_regime(df) == "unknown"


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
    assert classify_regime(df) == "trending"

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
    assert classify_regime(df, config_path=str(custom_cfg)) != "trending"


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

    assert (
        classify_regime(df_low, df_high, config_path=str(cfg))
        == "trending"
    )


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

    assert (
        classify_regime(df_low, df_high, config_path=str(cfg))
        != "trending"
    )


def test_classify_regime_async_matches_sync():
    df = _make_trending_df()
    sync_res = classify_regime(df)

    async def run():
        return await classify_regime_async(df)

    async_res = asyncio.run(run())
    assert async_res == sync_res


def test_analyze_symbol_async_consistent():
    df = _make_trending_df()
    regime = classify_regime(df)
    strategy = strategy_for(regime)
    sync_score, sync_dir = asyncio.run(evaluate_async(strategy, df, {}))

    async def run():
        return await analyze_symbol("AAA", df, "cex", {}, None)

    res = asyncio.run(run())
    assert res["regime"] == regime
    assert res["score"] == sync_score
    assert res["direction"] == sync_dir

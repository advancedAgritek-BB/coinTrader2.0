import pandas as pd
import pytest

import importlib.util
import pathlib
import types

path = pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "strategy" / "micro_scalp_bot.py"
spec = importlib.util.spec_from_file_location("micro_scalp_bot", path)
micro_scalp_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(micro_scalp_bot)


@pytest.fixture
def make_df():
    def _factory(prices, volumes, opens=None, highs=None, lows=None):
        opens = prices if opens is None else opens
        highs = [p + 0.5 for p in prices] if highs is None else highs
        lows = [p - 0.5 for p in prices] if lows is None else lows
        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )

    return _factory


def test_micro_scalp_long_signal(make_df):
    prices = list(range(1, 21))
    volumes = [100] * 19 + [150]
    df = make_df(prices, volumes)
    score, direction = micro_scalp_bot.generate_signal(df, None)
    assert direction == "long"
    assert 0 < score <= 1


def test_cross_with_momentum_and_wick(make_df):
    prices = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9]
    volumes = [100] * len(prices)
    cfg = {
        "micro_scalp_bot": {
            "ema_fast": 3,
            "ema_slow": 8,
            "lower_wick_pct": 0.3,
            "min_momentum_pct": 0.02,
            "fresh_cross_only": False,
            "min_vol_z": 0,
        }
    }
    df = make_df(prices, volumes)
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_volume_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [1] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"min_vol_z": 2, "volume_window": 5, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_atr_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"atr_period": 3, "min_atr_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_blocks_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(20, 9, -1))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp_bot": {"trend_fast": 3, "trend_slow": 5, "fresh_cross_only": False, "min_vol_z": 0}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_allows_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(10, 21))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp_bot": {"trend_fast": 3, "trend_slow": 5, "fresh_cross_only": False, "min_vol_z": 0}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert direction == "long"
    assert score > 0


def test_min_momentum_blocks_signal(make_df):
    prices = [10 + i * 0.01 for i in range(10)]
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"min_momentum_pct": 0.01, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_confirm_bars_blocks_fresh_cross(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"confirm_bars": 2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_fresh_cross_only_signal(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"fresh_cross_only": True, "confirm_bars": 1, "min_vol_z": 0}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_fresh_cross_only_requires_change(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"fresh_cross_only": True, "confirm_bars": 1}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_long(make_df):
    prices = list(range(1, 11))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "low"] = df["close"].iloc[-1] - 0.05
    cfg = {"micro_scalp_bot": {"wick_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_short(make_df):
    prices = list(range(10, 0, -1))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] + 0.05
    cfg = {"micro_scalp_bot": {"wick_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


@pytest.mark.parametrize(
    "prices,volumes,cfg",
    [
        (
            [10 + i * 0.01 for i in range(10)],
            [100] * 10,
            {"micro_scalp_bot": {"min_momentum_pct": 0.01, "fresh_cross_only": False}},
        ),
        (
            list(range(1, 11)),
            [1] * 10,
            {"micro_scalp_bot": {"min_vol_z": 2, "volume_window": 5, "fresh_cross_only": False}},
        ),
        (
            list(range(1, 11)),
            [100] * 10,
            {"micro_scalp_bot": {"atr_period": 3, "min_atr_pct": 0.2, "fresh_cross_only": False}},
        ),
    ],
)
def test_filters_return_none(make_df, prices, volumes, cfg):
    df = make_df(prices, volumes)
    assert micro_scalp_bot.generate_signal(df, cfg) == (0.0, "none")


class DummyMempool:
    async def is_suspicious(self, threshold):
        return True


def test_mempool_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"fresh_cross_only": False, "min_vol_z": 0}}
    monitor = DummyMempool()
    score, direction = micro_scalp_bot.generate_signal(
        df,
        cfg,
        mempool_monitor=monitor,
        mempool_cfg={"enabled": True},
    )
    assert (score, direction) == (0.0, "none")


class LowFeeMempool:
    async def fetch_priority_fee(self):
        return 3.0

    async def is_suspicious(self, threshold):
        return False


def test_mempool_fee_boosts_score(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    base_score, base_dir = micro_scalp_bot.generate_signal(df, {"micro_scalp_bot": {"min_vol_z": 0}})
    monitor = LowFeeMempool()
    boosted, boosted_dir = micro_scalp_bot.generate_signal(
        df, {"micro_scalp_bot": {"min_vol_z": 0}}, mempool_monitor=monitor
    )
    assert boosted_dir == base_dir
    assert boosted == pytest.approx(base_score * 1.2)


def test_tick_data_extends(make_df):
    df = make_df([1, 2, 3], [100, 100, 100])
    tick = make_df([4], [50])
    cfg = {"micro_scalp_bot": {"fresh_cross_only": False, "min_vol_z": 0}}
    micro_scalp_bot.generate_signal(df, cfg, tick_data=tick)


def test_trend_filter_disabled_allows_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(20, 9, -1))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {
        "micro_scalp_bot": {
            "trend_fast": 3,
            "trend_slow": 5,
            "fresh_cross_only": False,
            "trend_filter": False,
            "min_vol_z": 0,
        }
    }

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert direction == "long"
    assert score > 0


def test_imbalance_filter_disabled_allows_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    book = {"bids": [(1, 10)], "asks": [(1, 20)]}
    cfg = {
        "micro_scalp_bot": {
            "fresh_cross_only": False,
            "imbalance_ratio": 2,
            "imbalance_filter": False,
            "min_vol_z": 0,
        }
    }

    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert direction == "long"
    assert score > 0


def test_spread_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    book = {"bids": [(9.95, 1)], "asks": [(10.05, 1)]}
    cfg = {"micro_scalp_bot": {"fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert (score, direction) == (0.0, "none")

def test_spread_ratio_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    book = {"bids": [(10.0, 1)], "asks": [(10.1, 1)]}
    cfg = {"micro_scalp_bot": {"fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert (score, direction) == (0.0, "none")


def test_trainer_model_influence(make_df, monkeypatch):
    prices = list(range(1, 21))
    volumes = [100] * 19 + [150]
    df = make_df(prices, volumes)
    cfg = {"micro_scalp_bot": {"fresh_cross_only": False, "min_vol_z": 0}, "atr_normalization": False}
    monkeypatch.setattr(micro_scalp_bot, "MODEL", None)
    base, direction = micro_scalp_bot.generate_signal(df, cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.3)
    monkeypatch.setattr(micro_scalp_bot, "MODEL", dummy)
    score, direction2 = micro_scalp_bot.generate_signal(df, cfg)
    assert direction2 == direction
    assert score == pytest.approx((base + 0.3) / 2)

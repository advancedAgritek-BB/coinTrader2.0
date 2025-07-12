import pandas as pd
import pytest

import importlib.util
import pathlib
import types
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

crypto_bot = types.ModuleType("crypto_bot")
strategy_pkg = types.ModuleType("crypto_bot.strategy")
execution_pkg = types.ModuleType("crypto_bot.execution")
utils_pkg = types.ModuleType("crypto_bot.utils")
sys.modules.setdefault("crypto_bot", crypto_bot)
sys.modules.setdefault("crypto_bot.strategy", strategy_pkg)
sys.modules.setdefault("crypto_bot.execution", execution_pkg)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)

spec = importlib.util.spec_from_file_location(
    "crypto_bot.execution.solana_mempool",
    ROOT / "crypto_bot/execution/solana_mempool.py",
)
sol_mempool = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.execution.solana_mempool"] = sol_mempool
spec.loader.exec_module(sol_mempool)

# Minimal stubs for utility functions used by the strategy
ind_cache = types.ModuleType("crypto_bot.utils.indicator_cache")
ind_cache.cache_series = lambda *a, **k: a[2]
vol_mod = types.ModuleType("crypto_bot.utils.volatility")
vol_mod.normalize_score_by_volatility = lambda _df, s: s
vol_filter = types.ModuleType("crypto_bot.volatility_filter")
vol_filter.calc_atr = lambda df, window=14: df["high"].sub(df["low"]).rolling(window).mean().iloc[-1]
sys.modules["crypto_bot.utils.indicator_cache"] = ind_cache
sys.modules["crypto_bot.utils.volatility"] = vol_mod
sys.modules["crypto_bot.volatility_filter"] = vol_filter

for mod_name in ["indicator_cache", "volatility"]:
    m_spec = importlib.util.spec_from_file_location(
        f"crypto_bot.utils.{mod_name}", ROOT / f"crypto_bot/utils/{mod_name}.py"
    )
    module = importlib.util.module_from_spec(m_spec)
    sys.modules[f"crypto_bot.utils.{mod_name}"] = module
    m_spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location(
    "crypto_bot.strategy.micro_scalp_bot",
    ROOT / "crypto_bot/strategy/micro_scalp_bot.py",
)
micro_scalp_bot = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.strategy.micro_scalp_bot"] = micro_scalp_bot
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
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert 0 < score <= 1


def test_cross_with_momentum_and_wick(make_df):
    prices = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9]
    volumes = [100] * len(prices)
    cfg = {"micro_scalp": {"ema_fast": 3, "ema_slow": 8, "lower_wick_pct": 0.3, "min_momentum_pct": 0.02}}
    df = make_df(prices, volumes)
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_volume_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [1] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"min_vol_z": 2, "volume_window": 5}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_atr_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"atr_period": 3, "min_atr_pct": 0.2}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_blocks_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(20, 9, -1))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_allows_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(10, 21))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5, "fresh_cross_only": False}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert direction == "long"
    assert score > 0


def test_min_momentum_blocks_signal(make_df):
    prices = [10 + i * 0.01 for i in range(10)]
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"min_momentum_pct": 0.01}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_confirm_bars_blocks_fresh_cross(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"confirm_bars": 2}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_fresh_cross_only_signal(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True, "confirm_bars": 1}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_fresh_cross_only_requires_change(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_long(make_df):
    prices = list(range(1, 11))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "low"] = df["close"].iloc[-1] - 0.05
    cfg = {"micro_scalp": {"wick_pct": 0.2}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_short(make_df):
    prices = list(range(10, 0, -1))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] + 0.05
    cfg = {"micro_scalp": {"wick_pct": 0.2}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_volume_std_zero_allows_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"min_vol_z": 0, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_book_imbalance_blocks_long(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    book = {"bids": [[10, 1.0]], "asks": [[10, 5.0]]}
    cfg = {"micro_scalp": {"imbalance_ratio": 2.0, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert (score, direction) == (0.0, "none")


def test_book_imbalance_blocks_short(make_df):
    prices = list(range(10, 0, -1))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    book = {"bids": [[10, 5.0]], "asks": [[10, 1.0]]}
    cfg = {"micro_scalp": {"imbalance_ratio": 2.0, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert (score, direction) == (0.0, "none")


def test_book_imbalance_penalty_reduces_score(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    base_score, base_dir = micro_scalp_bot.generate_signal(
        df, {"micro_scalp": {"fresh_cross_only": False}}
    )
    book = {"bids": [[10, 1.0]], "asks": [[10, 5.0]]}
    cfg = {
        "micro_scalp": {
            "imbalance_ratio": 2.0,
            "imbalance_penalty": 0.5,
            "fresh_cross_only": False,
        }
    }
    score, direction = micro_scalp_bot.generate_signal(df, cfg, book=book)
    assert direction == base_dir
    assert 0 < score < base_score


@pytest.mark.parametrize(
    "prices,volumes,cfg",
    [
        ([10 + i * 0.01 for i in range(10)], [100] * 10, {"micro_scalp": {"min_momentum_pct": 0.01}}),
        (list(range(1, 11)), [1] * 10, {"micro_scalp": {"min_vol_z": 2, "volume_window": 5}}),
        (list(range(1, 11)), [100] * 10, {"micro_scalp": {"atr_period": 3, "min_atr_pct": 0.2}}),
    ],
)
def test_filters_return_none(make_df, prices, volumes, cfg):
    df = make_df(prices, volumes)
    assert micro_scalp_bot.generate_signal(df, cfg) == (0.0, "none")


def test_mempool_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {
        "micro_scalp": {"fresh_cross_only": False},
        "mempool_monitor": {"enabled": True, "suspicious_fee_threshold": 100},
    }

    class DummyMonitor:
        def is_suspicious(self, threshold):
            assert threshold == 100
            return True

    score, direction = micro_scalp_bot.generate_signal(
        df,
        cfg,
        mempool_monitor=DummyMonitor(),
        mempool_cfg=cfg["mempool_monitor"],
    )
    assert (score, direction) == (0.0, "none")

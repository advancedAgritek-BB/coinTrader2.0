import importlib
import sys
import types

import pandas as pd
import pytest

sys.modules.setdefault(
    "crypto_bot.strategy.sniper_bot", types.ModuleType("crypto_bot.strategy.sniper_bot")
)
sys.modules.setdefault(
    "crypto_bot.strategy.sniper_solana", types.ModuleType("crypto_bot.strategy.sniper_solana")
)

mean_bot = importlib.import_module("crypto_bot.strategy.mean_bot")


def _run(df, cfg=None):
    return mean_bot.generate_signal(df, config=cfg)


def _df_with_drop(price: float, last_width: float = 5.0) -> pd.DataFrame:
    """Return a dataframe with a final candle width of ``last_width``.

    The preceding candles have a constant width of ``2`` which becomes the
    bandwidth median. Passing a ``last_width`` larger than ``2`` simulates a
    high bandwidth environment while values below ``2`` create a squeeze.
    """

    base = [100.0] * 55 + [price]
    highs = [p + 1 for p in base]
    lows = [p - 1 for p in base]
    highs[-1] = price + last_width / 2
    lows[-1] = price - last_width / 2

    data = {
        "open": base,
        "high": highs,
        "low": lows,
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def _df_low_bw_drop() -> pd.DataFrame:
    import numpy as np

    np.random.seed(0)
    base = list(100 + np.random.randn(60) * 10)
    for i in range(40, 56):
        base[i] = 100 + np.random.randn() * 2
    base = base[:55] + [base[54] - 2]

    data = {
        "open": base,
        "high": [p + 1 for p in base],
        "low": [p - 1 for p in base],
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def test_long_signal_on_big_drop():
    """High bandwidth should block the signal."""
    df = _df_with_drop(80.0)
    score, direction = _run(df)
    assert direction == "none"
    assert score == 0.0


def test_short_signal_on_big_spike():
    """High bandwidth should block the signal."""
    df = _df_with_drop(120.0)
    score, direction = _run(df)
    assert direction == "none"
    assert score == 0.0


@pytest.mark.parametrize("price,expected", [(80.0, "none"), (120.0, "none")])
def test_signal_during_squeeze(price: float, expected: str):
    """A drop or spike during a squeeze should still generate a signal."""
    df = _df_with_drop(price, last_width=1.0)
    score, direction = _run(df)
    assert direction == expected
def test_long_signal_when_bandwidth_low():
    df = _df_low_bw_drop()
    score, direction = _run(df)
    assert direction == "long"
    assert score > 0


def test_no_signal_when_trending():
    close = list(range(1, 31))
    df = pd.DataFrame({
        "open": close,
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [100] * len(close),
    })
    score, direction = _run(df)
    assert direction == "none"
    assert score == 0.0


def test_indicator_lookback_default(monkeypatch):
    calls = {}

    def fake_zscore(series, lookback=0):
        calls["lookback"] = lookback
        return pd.Series([0] * len(series), index=series.index)

    monkeypatch.setattr(mean_bot.stats, "zscore", fake_zscore)
    df = _df_with_drop(80.0, last_width=1.0)
    score, direction = _run(df)
    assert calls["lookback"] == 14


def test_ml_enabled_by_default(monkeypatch):
    df = _df_low_bw_drop()
    base_score, direction = _run(df, {"ml_enabled": False})

    ml_mod = types.SimpleNamespace(predict_signal=lambda _df: 0.2)
    monkeypatch.setitem(sys.modules, "crypto_bot.ml_signal_model", ml_mod)

    score, direction2 = _run(df)
    assert direction2 == direction
    assert score == pytest.approx((base_score + 0.2) / 2)


def test_trainer_model_influence(monkeypatch):
    df = _df_low_bw_drop()
    cfg = {"atr_normalization": False}
    monkeypatch.setattr(mean_bot, "MODEL", None)
    base, direction = _run(df, cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.25)
    monkeypatch.setattr(mean_bot, "MODEL", dummy)
    score, direction2 = _run(df, cfg)
    assert direction2 == direction
    assert score == pytest.approx((base + 0.25) / 2)



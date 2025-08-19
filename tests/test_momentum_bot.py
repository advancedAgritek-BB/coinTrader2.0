import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "momentum_bot",
    Path(__file__).resolve().parents[1]
    / "crypto_bot"
    / "strategy"
    / "momentum_bot.py",
)
momentum_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(momentum_bot)


def _make_df(length: int = 30) -> pd.DataFrame:
    prices = list(range(length))
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [100] * length,
        }
    )


def test_long_signal_macd(monkeypatch):
    df = _make_df()
    monkeypatch.setattr(
        momentum_bot.ta.trend,
        "macd",
        lambda *args, **kwargs: pd.Series([1] * len(df)),
    )
    monkeypatch.setattr(
        momentum_bot.ta.momentum,
        "rsi",
        lambda *args, **kwargs: pd.Series([30] * len(df)),
    )
    score, direction = momentum_bot.generate_signal(
        df, config={"atr_normalization": False}
    )
    assert direction == "long"
    assert score == pytest.approx(0.8)


def test_long_signal_rsi(monkeypatch):
    df = _make_df()
    monkeypatch.setattr(
        momentum_bot.ta.trend,
        "macd",
        lambda *args, **kwargs: pd.Series([-1] * len(df)),
    )
    monkeypatch.setattr(
        momentum_bot.ta.momentum,
        "rsi",
        lambda *args, **kwargs: pd.Series([60] * len(df)),
    )
    score, direction = momentum_bot.generate_signal(
        df, config={"atr_normalization": False}
    )
    assert direction == "long"
    assert score == pytest.approx(0.8)

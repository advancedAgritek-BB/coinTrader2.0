import pytest
pytest.importorskip("pandas")
import importlib.util
import pandas as pd
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "solana_scalping",
    Path(__file__).resolve().parents[1] / "crypto_bot/strategy/solana_scalping.py",
)
solana_scalping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solana_scalping)


def make_df(prices):
    return pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1]*len(prices),
    })


def test_long_signal():
    prices = list(range(60, 30, -1)) + [31, 32, 33, 34]
    df = make_df(prices)
    score, direction = solana_scalping.generate_signal(df, {})
    assert direction == "long"
    assert score > 0


def test_short_signal():
    prices = list(range(30, 61)) + [60, 59, 58, 57]
    df = make_df(prices)
    score, direction = solana_scalping.generate_signal(df, {})
    assert direction == "short"
    assert score > 0


def test_no_signal_when_conditions_not_met():
    prices = list(range(1, 40))
    df = make_df(prices)
    score, direction = solana_scalping.generate_signal(df, {})
    assert (score, direction) == (0.0, "none")

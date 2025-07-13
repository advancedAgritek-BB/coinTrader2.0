import pytest
pytest.importorskip("pandas")
import pandas as pd
from crypto_bot.strategy import pair_arbitrage, grid_bot
from crypto_bot.strategy.grid_bot import GridConfig


def _df(price: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [price], "high": [price], "low": [price], "volume": [1.0]})


def test_pair_arbitrage_triggers_grid(monkeypatch):
    called = []

    def fake_grid(df, config=None):
        called.append(df["close"].iloc[-1])
        return 0.5, "long"

    monkeypatch.setattr(grid_bot, "generate_signal", fake_grid)

    df_map = {"AAA": _df(110.0), "BBB": _df(100.0)}
    cfg = GridConfig(arbitrage_pairs=[("AAA", "BBB")], arbitrage_threshold=0.05)
    signals = pair_arbitrage.generate_signals(df_map, cfg)

    assert len(called) == 2
    assert signals == [("AAA", 0.5, "long"), ("BBB", 0.5, "long")]


def test_pair_arbitrage_no_op(monkeypatch):
    called = []

    def fake_grid(df, config=None):
        called.append(True)
        return 0.5, "long"

    monkeypatch.setattr(grid_bot, "generate_signal", fake_grid)

    df_map = {"AAA": _df(100.0), "BBB": _df(100.5)}
    cfg = GridConfig(arbitrage_pairs=[("AAA", "BBB")], arbitrage_threshold=0.05)
    signals = pair_arbitrage.generate_signals(df_map, cfg)

    assert not called
    assert signals == []

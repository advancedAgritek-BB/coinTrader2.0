import pandas as pd
import ccxt

from crypto_bot.backtest import backtest_runner


class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        ts = pd.date_range("2020-01-01", periods=120, freq="H")
        price = 100.0
        data = []
        for t in ts[:limit]:
            price += 0.5
            data.append([int(t.timestamp() * 1000), price, price + 1, price - 1, price, 100])
        return data


def _constant_strategy(df):
    return 1.0, "long"


def test_backtest_tracks_switches(monkeypatch):
    monkeypatch.setattr(ccxt, "binance", lambda: FakeExchange())
    regimes = ["trending"] * 40 + ["sideways"] * 40 + ["trending"] * 40

    def fake_classify(df):
        idx = min(len(df) - 1, len(regimes) - 1)
        return regimes[idx], {regimes[idx]: 1.0}
        return regimes[idx], {}

    monkeypatch.setattr(backtest_runner, "classify_regime", fake_classify)
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)

    result = backtest_runner.backtest(
        "BTC/USDT",
        "1h",
        since=0,
        limit=120,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        slippage_pct=0.01,
        fee_pct=0.0,
        seed=1,
    )
    row = result.iloc[0]
    assert row["switches"] > 0
    assert row["slippage_cost"] > 0


def test_backtest_misclassification(monkeypatch):
    monkeypatch.setattr(ccxt, "binance", lambda: FakeExchange())
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {"trending": 1.0}))
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {}))
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)

    result = backtest_runner.backtest(
        "BTC/USDT",
        "1h",
        since=0,
        limit=80,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        misclass_prob=1.0,
        seed=42,
    )
    row = result.iloc[0]
    assert row["misclassified"] > 0


def test_walk_forward_optimize(monkeypatch):
    monkeypatch.setattr(ccxt, "binance", lambda: FakeExchange())
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {"trending": 1.0}))
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {}))
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)

    df = backtest_runner.walk_forward_optimize(
        "BTC/USDT",
        "1h",
        since=0,
        limit=60,
        window=20,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        seed=3,
    )
    assert {"regime", "train_stop_loss_pct", "train_take_profit_pct"} <= set(df.columns)

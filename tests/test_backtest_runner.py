import pandas as pd
import ccxt
import logging

from crypto_bot.backtest import backtest_runner as bt, backtest_runner
from crypto_bot.backtest.backtest_runner import BacktestConfig, BacktestRunner


class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        ts = pd.date_range("2020-01-01", periods=120, freq="H")
        price = 100.0
        data = []
        for t in ts[:limit]:
            price += 0.5
            data.append(
                [int(t.timestamp() * 1000), price, price + 1, price - 1, price, 100]
            )
        return data


def _constant_strategy(df):
    return 1.0, "long"


def test_backtest_tracks_switches(monkeypatch):
    regimes = ["trending"] * 40 + ["sideways"] * 40 + ["trending"] * 40

    def fake_classify(df):
        idx = min(len(df) - 1, len(regimes) - 1)
        return regimes[idx], {regimes[idx]: 1.0}

    monkeypatch.setattr(bt, "classify_regime", fake_classify)
    monkeypatch.setattr(bt, "strategy_for", lambda r: _constant_strategy)
    fake_data = FakeExchange().fetch_ohlcv("XBT/USDT", "1h", 0, 120)
    monkeypatch.setattr(BacktestRunner, "_fetch_data", lambda self: fake_data)
    monkeypatch.setattr(backtest_runner, "classify_regime", fake_classify)
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)

    cfg = BacktestConfig(
        symbol="XBT/USDT",
        timeframe="1h",
        since=0,
        limit=120,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        slippage_pct=0.01,
        fee_pct=0.0,
        seed=1,
    )
    runner = BacktestRunner(cfg)
    df = runner.run_grid()
    row = df.iloc[0]
    assert row["switches"] > 0
    assert row["slippage_cost"] > 0


def test_backtest_misclassification(monkeypatch):
    monkeypatch.setattr(ccxt, "binance", lambda: FakeExchange())
    monkeypatch.setattr(bt, "classify_regime", lambda df: ("trending", {"trending": 1.0}))
    monkeypatch.setattr(bt, "strategy_for", lambda r: _constant_strategy)
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {}))
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)
    fake_data = FakeExchange().fetch_ohlcv("XBT/USDT", "1h", 0, 80)
    monkeypatch.setattr(BacktestRunner, "_fetch_data", lambda self: fake_data)

    cfg = BacktestConfig(
        symbol="XBT/USDT",
        timeframe="1h",
        since=0,
        limit=80,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        misclass_prob=1.0,
        seed=42,
    )
    runner = BacktestRunner(cfg)
    df = runner.run_grid()
    row = df.iloc[0]
    assert row["misclassified"] > 0


def test_walk_forward_optimize(monkeypatch):
    monkeypatch.setattr(ccxt, "binance", lambda: FakeExchange())
    monkeypatch.setattr(bt, "classify_regime", lambda df: ("trending", {"trending": 1.0}))
    monkeypatch.setattr(bt, "strategy_for", lambda r: _constant_strategy)
    monkeypatch.setattr(backtest_runner, "classify_regime", lambda df: ("trending", {}))
    monkeypatch.setattr(backtest_runner, "strategy_for", lambda r: _constant_strategy)
    fake_data = FakeExchange().fetch_ohlcv("XBT/USDT", "1h", 0, 60)
    monkeypatch.setattr(BacktestRunner, "_fetch_data", lambda self: fake_data)

    cfg = BacktestConfig(
        symbol="XBT/USDT",
        timeframe="1h",
        since=0,
        limit=60,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
        seed=3,
        window=20,
    )
    runner = BacktestRunner(cfg)
    df = runner.run_walk_forward()
    assert {"regime", "train_stop_loss_pct", "train_take_profit_pct"} <= set(df.columns)


def test_skip_sniper_solana(caplog):
    caplog.set_level(logging.INFO)
    df = bt.backtest(
        symbol="XBT/USDT",
        timeframe="1h",
        since=0,
        limit=10,
        strategy="sniper_solana",
    )
    assert df.empty
    assert "Sniper Solana not backtestable" in caplog.text


def test_hft_threshold_env(monkeypatch):
    cfg = BacktestConfig(
        symbol="XBT/USDT",
        timeframe="30s",
        since=0,
        limit=1,
    )

    df_prepared = pd.DataFrame(
        {
            "adx": [20],
            "ema_fast": [101],
            "ema_slow": [100],
            "bb_width": [0.05],
            "volume": [100],
            "volume_ma": [100],
            "rsi": [80],
            "close": [100],
            "normalized_range": [1.0],
        }
    )

    monkeypatch.setattr(BacktestRunner, "_prepare_data", lambda self, df: df)
    runner = BacktestRunner(cfg, df=df_prepared)

    regimes = runner._precompute_regimes(df_prepared)
    assert regimes[-1] == "trending"

    monkeypatch.setenv("HFT_ADX_TRENDING_MIN", "50")
    regimes = runner._precompute_regimes(df_prepared)
    assert regimes[-1] == "sideways"

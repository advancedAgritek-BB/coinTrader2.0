import pandas as pd
import numpy as np
import ccxt

from crypto_bot.backtest import backtest_runner
from crypto_bot.strategy import trend_bot

class FakeExchange:
    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        ts = pd.date_range('2020-01-01', periods=120, freq='H')
        price = 100.0
        data = []
        for t in ts:
            price += 0.5
            data.append([int(t.timestamp() * 1000), price, price + 1, price - 1, price, 100])
        return data

def test_backtest_returns_metrics(monkeypatch):
    monkeypatch.setattr(ccxt, 'binance', lambda: FakeExchange())
    monkeypatch.setattr(backtest_runner, 'classify_regime', lambda df: 'trending')
    monkeypatch.setattr(backtest_runner, 'route', lambda regime, mode: trend_bot.generate_signal)

    result = backtest_runner.backtest(
        'BTC/USDT',
        '1h',
        since=0,
        limit=120,
        stop_loss_range=[0.01],
        take_profit_range=[0.02],
    )

    assert {'pnl', 'max_drawdown', 'sharpe', 'stop_loss_pct', 'take_profit_pct'} <= set(result.columns)

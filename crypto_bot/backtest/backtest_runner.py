import ccxt
import pandas as pd
from typing import Dict

from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import route
from crypto_bot.signals.signal_scoring import evaluate


def backtest(symbol: str, timeframe: str, since: int, limit: int = 1000, mode: str = 'cex') -> pd.DataFrame:
    """Run a simple regime aware backtest."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    results = []
    for i in range(60, len(df)):
        subset = df.iloc[:i]
        regime = classify_regime(subset)
        strategy_fn = route(regime, mode)
        score, direction = evaluate(strategy_fn, subset)
        results.append({'timestamp': subset['timestamp'].iloc[-1], 'regime': regime, 'score': score, 'direction': direction})
    return pd.DataFrame(results)

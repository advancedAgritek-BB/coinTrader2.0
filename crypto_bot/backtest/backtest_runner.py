import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Iterable

from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import route
from crypto_bot.signals.signal_scoring import evaluate


def _run_single(df: pd.DataFrame, stop_loss: float, take_profit: float, mode: str) -> Dict:
    """Execute a naive backtest for one parameter set."""
    position = None
    entry_price = 0.0
    equity = 1.0
    peak_equity = 1.0
    max_dd = 0.0
    returns: List[float] = []

    for i in range(60, len(df)):
        subset = df.iloc[: i + 1]
        regime = classify_regime(subset)
        strategy_fn = route(regime, mode)
        score, direction = evaluate(strategy_fn, subset)
        price = subset['close'].iloc[-1]

        if position is None and direction in {'long', 'short'}:
            position = direction
            entry_price = price
            continue

        if position is not None:
            change = (
                (price - entry_price) / entry_price
                if position == 'long'
                else (entry_price - price) / entry_price
            )
            if change <= -stop_loss or change >= take_profit:
                equity *= 1 + change
                returns.append(change)
                peak_equity = max(peak_equity, equity)
                dd = 1 - equity / peak_equity
                max_dd = max(max_dd, dd)
                position = None
                entry_price = 0.0

    if position is not None:
        final_price = df['close'].iloc[-1]
        change = (
            (final_price - entry_price) / entry_price
            if position == 'long'
            else (entry_price - final_price) / entry_price
        )
        equity *= 1 + change
        returns.append(change)
        peak_equity = max(peak_equity, equity)
        dd = 1 - equity / peak_equity
        max_dd = max(max_dd, dd)

    pnl = equity - 1
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))

    return {
        'stop_loss_pct': stop_loss,
        'take_profit_pct': take_profit,
        'pnl': pnl,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
    }


def backtest(
    symbol: str,
    timeframe: str,
    since: int,
    limit: int = 1000,
    mode: str = 'cex',
    stop_loss_range: Iterable[float] | None = None,
    take_profit_range: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Run a regime aware backtest and evaluate parameter combinations."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(
        symbol, timeframe=timeframe, since=since, limit=limit
    )
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    stop_loss_range = stop_loss_range or [0.02]
    take_profit_range = take_profit_range or [0.04]

    results = []
    for sl in stop_loss_range:
        for tp in take_profit_range:
            metrics = _run_single(df, sl, tp, mode)
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False).reset_index(drop=True)
    return results_df

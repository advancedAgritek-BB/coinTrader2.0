import ccxt
import pandas as pd
import numpy as np
from typing import Dict, Iterable, List, Optional
from numpy.random import default_rng, Generator

from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import strategy_for
from crypto_bot.signals.signal_scoring import evaluate

# Available regimes used when simulating misclassification
_REGIMES = [
    "trending",
    "sideways",
    "mean-reverting",
    "breakout",
    "volatile",
]


def _trade_return(
    entry: float,
    exit: float,
    position: str,
    cost: float,
) -> float:
    """Calculate trade return after cost."""
    raw = (exit - entry) / entry if position == "long" else (entry - exit) / entry
    return raw - cost


def _run_single(
    df: pd.DataFrame,
    stop_loss: float,
    take_profit: float,
    mode: str,
    slippage_pct: float,
    fee_pct: float,
    misclass_prob: float,
    rng: Generator,
) -> Dict:
    """Execute a regime aware backtest for one parameter set."""

    position: Optional[str] = None
    entry_price = 0.0
    equity = 1.0
    peak_equity = 1.0
    max_dd = 0.0
    returns: List[float] = []
    switches = 0
    misclassified = 0
    slippage_cost = 0.0

    last_regime: Optional[str] = None

    for i in range(60, len(df)):
        subset = df.iloc[: i + 1]
        true_regime = classify_regime(subset)
        regime = true_regime
        if misclass_prob > 0 and rng.random() < misclass_prob:
            regime = rng.choice([r for r in _REGIMES if r != true_regime])
            misclassified += 1

        if last_regime is not None and regime != last_regime and position is not None:
            exit_price = subset["close"].iloc[-1]
            cost = (fee_pct + slippage_pct) * 2
            r = _trade_return(entry_price, exit_price, position, cost)
            equity *= 1 + r
            returns.append(r)
            slippage_cost += cost
            peak_equity = max(peak_equity, equity)
            max_dd = max(max_dd, 1 - equity / peak_equity)
            position = None
            entry_price = 0.0
            switches += 1
        last_regime = regime

        strategy_fn = strategy_for(regime)
        _, direction = evaluate(strategy_fn, subset, None)
        price = subset["close"].iloc[-1]

        if position is None and direction in {"long", "short"}:
            position = direction
            entry_price = price
            continue

        if position is not None:
            change = (
                (price - entry_price) / entry_price
                if position == "long"
                else (entry_price - price) / entry_price
            )
            if change <= -stop_loss or change >= take_profit:
                exit_price = price
                cost = (fee_pct + slippage_pct) * 2
                r = _trade_return(entry_price, exit_price, position, cost)
                equity *= 1 + r
                returns.append(r)
                slippage_cost += cost
                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, 1 - equity / peak_equity)
                position = None
                entry_price = 0.0

    if position is not None:
        final_price = df["close"].iloc[-1]
        cost = (fee_pct + slippage_pct) * 2
        r = _trade_return(entry_price, final_price, position, cost)
        equity *= 1 + r
        returns.append(r)
        slippage_cost += cost
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, 1 - equity / peak_equity)

    pnl = equity - 1
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))

    return {
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
        "pnl": pnl,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "misclassified": misclassified,
        "switches": switches,
        "slippage_cost": slippage_cost,
    }


def backtest(
    symbol: str,
    timeframe: str,
    *,
    since: int,
    limit: int = 1000,
    mode: str = "cex",
    stop_loss_range: Iterable[float] | None = None,
    take_profit_range: Iterable[float] | None = None,
    slippage_pct: float = 0.001,
    fee_pct: float = 0.001,
    misclass_prob: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run regime-aware backtests over parameter ranges."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    stop_loss_range = stop_loss_range or [0.02]
    take_profit_range = take_profit_range or [0.04]
    rng = default_rng(seed)

    results = []
    for sl in stop_loss_range:
        for tp in take_profit_range:
            metrics = _run_single(
                df,
                sl,
                tp,
                mode,
                slippage_pct,
                fee_pct,
                misclass_prob,
                rng,
            )
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return results_df


def walk_forward_optimize(
    symbol: str,
    timeframe: str,
    *,
    since: int,
    limit: int,
    window: int,
    mode: str = "cex",
    stop_loss_range: Iterable[float] | None = None,
    take_profit_range: Iterable[float] | None = None,
    slippage_pct: float = 0.001,
    fee_pct: float = 0.001,
    misclass_prob: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Perform walk-forward optimization segmented by regime."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    stop_loss_range = stop_loss_range or [0.02]
    take_profit_range = take_profit_range or [0.04]
    rng = default_rng(seed)

    results: List[Dict] = []
    start = 0
    while start + window * 2 <= len(df):
        train = df.iloc[start : start + window]
        test = df.iloc[start + window : start + window * 2]
        regime = classify_regime(train)
        best_sl = stop_loss_range[0]
        best_tp = take_profit_range[0]
        best_sharpe = -np.inf
        for sl in stop_loss_range:
            for tp in take_profit_range:
                metrics = _run_single(
                    train,
                    sl,
                    tp,
                    mode,
                    slippage_pct,
                    fee_pct,
                    misclass_prob,
                    rng,
                )
                if metrics["sharpe"] > best_sharpe:
                    best_sharpe = metrics["sharpe"]
                    best_sl = sl
                    best_tp = tp
        test_metrics = _run_single(
            test,
            best_sl,
            best_tp,
            mode,
            slippage_pct,
            fee_pct,
            misclass_prob,
            rng,
        )
        test_metrics.update(
            {
                "regime": regime,
                "train_stop_loss_pct": best_sl,
                "train_take_profit_pct": best_tp,
            }
        )
        results.append(test_metrics)
        start += window

    return pd.DataFrame(results)

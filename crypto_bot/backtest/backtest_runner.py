import ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from numpy.random import default_rng, Generator

import ta
from crypto_bot.regime.regime_classifier import classify_regime, CONFIG
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


def _precompute_regimes(df: pd.DataFrame) -> List[str]:
    """Vectorized regime classification for each row."""
    # If ``classify_regime`` was monkeypatched (e.g. in tests), fall back to
    # repeatedly calling it to preserve expected behaviour.
    if classify_regime.__module__ != "crypto_bot.regime.regime_classifier":
        return [classify_regime(df.iloc[: i + 1])[0] for i in range(len(df))]

    cfg = CONFIG
    work = df.copy()

    work["ema_fast"] = ta.trend.ema_indicator(work["close"], window=cfg["ema_fast"])
    work["ema_slow"] = ta.trend.ema_indicator(work["close"], window=cfg["ema_slow"])
    work["adx"] = ta.trend.adx(
        work["high"], work["low"], work["close"], window=cfg["indicator_window"]
    )
    work["rsi"] = ta.momentum.rsi(work["close"], window=cfg["indicator_window"])
    work["atr"] = ta.volatility.average_true_range(
        work["high"], work["low"], work["close"], window=cfg["indicator_window"]
    )
    work["normalized_range"] = (work["high"] - work["low"]) / work["atr"]
    bb = ta.volatility.BollingerBands(work["close"], window=cfg["bb_window"])
    work["bb_width"] = bb.bollinger_wband()
    work["volume_ma"] = work["volume"].rolling(cfg["ma_window"]).mean()
    work["atr_ma"] = work["atr"].rolling(cfg["ma_window"]).mean()

    regimes: List[str] = []
    for i in range(len(work)):
        latest = work.iloc[i]
        volume_ma = latest["volume_ma"]
        atr_ma = latest["atr_ma"]
        trending = (
            latest["adx"] > cfg["adx_trending_min"]
            and latest["ema_fast"] > latest["ema_slow"]
        )

        if trending:
            regime = "trending"
        elif (
            latest["adx"] < cfg["adx_sideways_max"]
            and latest["bb_width"] < cfg["bb_width_sideways_max"]
        ):
            regime = "sideways"
        elif (
            latest["bb_width"] < cfg["bb_width_breakout_max"]
            and not pd.isna(volume_ma)
            and latest["volume"] > volume_ma * cfg["breakout_volume_mult"]
        ):
            regime = "breakout"
        elif (
            cfg["rsi_mean_rev_min"] <= latest["rsi"] <= cfg["rsi_mean_rev_max"]
            and abs(latest["close"] - latest["ema_fast"]) / latest["close"]
            < cfg["ema_distance_mean_rev_max"]
        ):
            regime = "mean-reverting"
        elif (
            not pd.isna(latest["normalized_range"])
            and latest["normalized_range"] > cfg["normalized_range_volatility_min"]
        ):
            regime = "volatile"
        else:
            regime = "sideways"
        regimes.append(regime)

    return regimes


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
    regimes = _precompute_regimes(df)

    for i in range(60, len(df)):
        subset = df.iloc[: i + 1]
        true_regime = regimes[i]
        regime = true_regime
        if misclass_prob > 0 and rng.random() < misclass_prob:
            regime = rng.choice([r for r in _REGIMES if r != true_regime])
            misclassified += 1

        if last_regime is not None and regime != last_regime and position is not None:
            exit_price = df["close"].iloc[i]
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
        _, direction, _ = evaluate(strategy_fn, subset, None)
        price = df["close"].iloc[i]

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
        regime, _ = classify_regime(train)
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


@dataclass
class BacktestConfig:
    """Configuration for running backtests."""

    symbol: str
    timeframe: str
    since: int
    limit: int = 1000
    mode: str = "cex"
    stop_loss_range: Iterable[float] | None = None
    take_profit_range: Iterable[float] | None = None
    slippage_pct: float = 0.001
    fee_pct: float = 0.001
    misclass_prob: float = 0.0
    seed: Optional[int] = None


class BacktestRunner:
    """Wrapper class providing a simple interface for grid backtests."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run_grid(self) -> pd.DataFrame:
        """Execute the standard grid search backtest."""
        return backtest(
            self.config.symbol,
            self.config.timeframe,
            since=self.config.since,
            limit=self.config.limit,
            mode=self.config.mode,
            stop_loss_range=self.config.stop_loss_range,
            take_profit_range=self.config.take_profit_range,
            slippage_pct=self.config.slippage_pct,
            fee_pct=self.config.fee_pct,
            misclass_prob=self.config.misclass_prob,
            seed=self.config.seed,
        )

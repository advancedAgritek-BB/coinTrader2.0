# pylint: disable=too-many-locals
"""Lightweight regime aware backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import asyncio
from typing import Dict, Iterable, List, Optional

import logging
import os
from crypto_bot.utils.market_loader import fetch_geckoterminal_ohlcv, timeframe_seconds

try:
    import ccxt.pro as ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.ModuleType("ccxt.pro")
import numpy as np
import pandas as pd
from numpy.random import default_rng, Generator

import ta

from crypto_bot.regime.regime_classifier import CONFIG, classify_regime
from crypto_bot.signals.signal_scoring import evaluate
from crypto_bot.strategy_router import strategy_for


logger = logging.getLogger(__name__)

_REGIMES = [
    "trending",
    "sideways",
    "mean-reverting",
    "breakout",
    "volatile",
]

# Fallback parameters used when :data:`CONFIG` is empty or missing keys.
_DEFAULT_CFG = {
    "ema_fast": 8,
    "ema_slow": 21,
    "indicator_window": 14,
    "bb_window": 20,
    "ma_window": 20,
    "adx_trending_min": 10,
    "adx_sideways_max": 18,
    "bb_width_sideways_max": 0.025,
    "bb_width_breakout_max": 4,
    "breakout_volume_mult": 1.5,
    "rsi_mean_rev_min": 30,
    "rsi_mean_rev_max": 70,
    "ema_distance_mean_rev_max": 0.02,
    "normalized_range_volatility_min": 1.5,
    "hft_adx_trending_min": 10,
    "hft_rsi_mean_rev_min": 30,
    "hft_rsi_mean_rev_max": 70,
}


@dataclass
class BacktestConfig:
    """Configuration for :class:`BacktestRunner`."""

    symbol: str
    timeframe: str
    since: int
    limit: int = 1000
    mode: str = "cex"
    stop_loss_range: Iterable[float] = field(default_factory=lambda: [0.02])
    take_profit_range: Iterable[float] = field(default_factory=lambda: [0.04])
    window: int = 50
    slippage_pct: float = 0.001
    fee_pct: float = 0.001
    misclass_prob: float = 0.0
    seed: Optional[int] = None
    risk_per_trade_pct: float = 0.01
    trailing_stop_atr_mult: float | None = None
    partial_tp_atr_mult: float | None = None


class BacktestRunner:
    """Execute regime aware backtests."""

    def __init__(
        self,
        config: BacktestConfig,
        exchange: ccxt.Exchange | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.exchange = exchange
        self.rng = np.random.default_rng(config.seed)
        df_raw: pd.DataFrame | list | None
        if df is not None:
            df_raw = df
        else:
            df_raw = self._fetch_data()
        if not isinstance(df_raw, pd.DataFrame):
            df_raw = pd.DataFrame(
                df_raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            if not df_raw.empty:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
        self.df_prepared = self._prepare_data(df_raw)

    @staticmethod
    def _build_cfg() -> Dict[str, float]:
        cfg = {**_DEFAULT_CFG, **CONFIG}
        for key in (
            "adx_trending_min",
            "rsi_mean_rev_min",
            "rsi_mean_rev_max",
            "hft_adx_trending_min",
            "hft_rsi_mean_rev_min",
            "hft_rsi_mean_rev_max",
        ):
            env_val = os.getenv(key.upper())
            if env_val is not None:
                try:
                    cfg[key] = float(env_val)
                except ValueError:
                    pass
        return cfg

    @staticmethod
    def _fetch_from_exchange(
        exchange: object,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> List[List[float]]:
        """Fetch OHLCV data handling async/sync exchanges."""
        result = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return result

    @staticmethod
    def _close_exchange(exchange: object) -> None:
        close = getattr(exchange, "close", None)
        if close is not None:
            res = close()
            if asyncio.iscoroutine(res):
                asyncio.run(res)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_fetch(symbol: str, timeframe: str, since: int, limit: int) -> pd.DataFrame:
        if symbol.endswith("/USDC"):
            res = asyncio.run(
                fetch_geckoterminal_ohlcv(symbol, timeframe=timeframe, limit=limit)
            )
            data = res[0] if isinstance(res, tuple) else res
            df = pd.DataFrame(
                data or [],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        exch = ccxt.binance()
        try:
            ohlcv = BacktestRunner._fetch_from_exchange(
                exch, symbol, timeframe, since, limit
            )
        finally:
            BacktestRunner._close_exchange(exch)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def _fetch_data(self) -> pd.DataFrame:
        if self.config.symbol.endswith("/USDC"):
            res = asyncio.run(
                fetch_geckoterminal_ohlcv(
                    self.config.symbol,
                    timeframe=self.config.timeframe,
                    limit=self.config.limit,
                )
            )
            data = res[0] if isinstance(res, tuple) else res
            df = pd.DataFrame(
                data or [],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        if self.exchange is None:
            return self._cached_fetch(
                self.config.symbol,
                self.config.timeframe,
                self.config.since,
                self.config.limit,
            )
        ohlcv = self._fetch_from_exchange(
            self.exchange,
            self.config.symbol,
            self.config.timeframe,
            self.config.since,
            self.config.limit,
        )
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._build_cfg()
        df = df.copy()
        df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg["ema_fast"])
        df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg["ema_slow"])
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=cfg["indicator_window"])
        df["rsi"] = ta.momentum.rsi(df["close"], window=cfg["indicator_window"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=cfg["indicator_window"]
        )
        df["normalized_range"] = (df["high"] - df["low"]) / df["atr"]
        bb = ta.volatility.BollingerBands(df["close"], window=cfg["bb_window"])
        df["bb_width"] = bb.bollinger_wband()
        df["volume_ma"] = df["volume"].rolling(cfg["ma_window"]).mean()
        df["atr_ma"] = df["atr"].rolling(cfg["ma_window"]).mean()
        return df

    def _precompute_regimes(self, df_prepared: pd.DataFrame) -> List[str]:
        if classify_regime.__module__ != "crypto_bot.regime.regime_classifier":
            return [classify_regime(df_prepared.iloc[: i + 1])[0] for i in range(len(df_prepared))]
        cfg = self._build_cfg()
        tf_sec = timeframe_seconds(None, self.config.timeframe)
        adx_min = cfg["adx_trending_min"]
        rsi_min = cfg["rsi_mean_rev_min"]
        rsi_max = cfg["rsi_mean_rev_max"]
        if tf_sec < 60:
            adx_min = cfg.get("hft_adx_trending_min", adx_min)
            rsi_min = cfg.get("hft_rsi_mean_rev_min", rsi_min)
            rsi_max = cfg.get("hft_rsi_mean_rev_max", rsi_max)

        trending = (df_prepared["adx"] > adx_min) & (
            df_prepared["ema_fast"] > df_prepared["ema_slow"]
        )
        sideways = (df_prepared["adx"] < cfg["adx_sideways_max"]) & (
            df_prepared["bb_width"] < cfg["bb_width_sideways_max"]
        )
        breakout = (df_prepared["bb_width"] < cfg["bb_width_breakout_max"]) & (
            df_prepared["volume"] > df_prepared["volume_ma"] * cfg["breakout_volume_mult"]
        )
        mean_rev = (
            df_prepared["rsi"].between(rsi_min, rsi_max)
            & (
                (df_prepared["close"] - df_prepared["ema_fast"]).abs()
                / df_prepared["close"]
                < cfg["ema_distance_mean_rev_max"]
            )
        )
        volatile = df_prepared["normalized_range"] > cfg["normalized_range_volatility_min"]

        regimes = np.select(
            [trending, sideways, breakout, mean_rev, volatile],
            ["trending", "sideways", "breakout", "mean-reverting", "volatile"],
            default="sideways",
        )
        return regimes.tolist()

    def _trade_return(self, entry: float, exit: float, position: str, cost: float) -> float:
        raw = (exit - entry) / entry if position == "long" else (entry - exit) / entry
        return raw - cost

    def _run_single(
        self,
        df_prepared: pd.DataFrame,
        stop_loss: float,
        take_profit: float,
        rng: np.random.Generator,
    ) -> dict:
        cfg = self.config
        position: Optional[str] = None
        entry_price = 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        returns: List[float] = []
        switches = 0
        misclassified = 0
        slippage_cost = 0.0
        position_size = 0.0
        trailing_stop = None
        partial_done = False

        last_regime: Optional[str] = None
        regimes = self._precompute_regimes(df_prepared)

        for i in range(60, len(df_prepared)):
            subset = df_prepared.iloc[: i + 1]
            true_regime = regimes[i]
            regime = true_regime
            if cfg.misclass_prob > 0 and rng.random() < cfg.misclass_prob:
                regime = rng.choice([r for r in _REGIMES if r != true_regime])
                misclassified += 1

            if last_regime is not None and regime != last_regime and position is not None:
                exit_price = df_prepared["close"].iloc[i]
                cost = (cfg.fee_pct + cfg.slippage_pct) * 2
                r = self._trade_return(entry_price, exit_price, position, cost)
                equity *= 1 + r * position_size
                returns.append(r)
                slippage_cost += cost
                peak = max(peak, equity)
                max_dd = max(max_dd, 1 - equity / peak)
                position = None
                entry_price = 0.0
                position_size = 0.0
                trailing_stop = None
                partial_done = False
                switches += 1
            last_regime = regime

            strategy_fn = strategy_for(regime)
            _, direction, _ = evaluate(strategy_fn, subset, None)
            price = df_prepared["close"].iloc[i]
            atr = df_prepared["atr"].iloc[i]

            if position is None and direction in {"long", "short"}:
                position = direction
                entry_price = price
                position_size = min(1.0, cfg.risk_per_trade_pct / stop_loss)
                if cfg.trailing_stop_atr_mult:
                    if position == "long":
                        trailing_stop = price - atr * cfg.trailing_stop_atr_mult
                    else:
                        trailing_stop = price + atr * cfg.trailing_stop_atr_mult
                continue

            if position is not None:
                change = (
                    (price - entry_price) / entry_price
                    if position == "long"
                    else (entry_price - price) / entry_price
                )
                if cfg.trailing_stop_atr_mult and trailing_stop is not None:
                    if position == "long":
                        trailing_stop = max(trailing_stop, price - atr * cfg.trailing_stop_atr_mult)
                        if price <= trailing_stop:
                            change = -stop_loss - 0.001  # force exit
                    else:
                        trailing_stop = min(trailing_stop, price + atr * cfg.trailing_stop_atr_mult)
                        if price >= trailing_stop:
                            change = -stop_loss - 0.001

                if (
                    cfg.partial_tp_atr_mult
                    and not partial_done
                    and position == "long"
                    and price >= entry_price + atr * cfg.partial_tp_atr_mult
                ) or (
                    cfg.partial_tp_atr_mult
                    and not partial_done
                    and position == "short"
                    and price <= entry_price - atr * cfg.partial_tp_atr_mult
                ):
                    exit_price = price
                    cost = (cfg.fee_pct + cfg.slippage_pct) * 2
                    r = self._trade_return(entry_price, exit_price, position, cost)
                    equity *= 1 + r * (position_size / 2)
                    returns.append(r)
                    slippage_cost += cost
                    position_size /= 2
                    partial_done = True

                if change <= -stop_loss or change >= take_profit:
                    exit_price = price
                    cost = (cfg.fee_pct + cfg.slippage_pct) * 2
                    r = self._trade_return(entry_price, exit_price, position, cost)
                    equity *= 1 + r * position_size
                    returns.append(r)
                    slippage_cost += cost
                    peak = max(peak, equity)
                    max_dd = max(max_dd, 1 - equity / peak)
                    position = None
                    entry_price = 0.0
                    position_size = 0.0
                    trailing_stop = None
                    partial_done = False

        if position is not None:
            final_price = df_prepared["close"].iloc[-1]
            cost = (cfg.fee_pct + cfg.slippage_pct) * 2
            r = self._trade_return(entry_price, final_price, position, cost)
            equity *= 1 + r * position_size
            returns.append(r)
            slippage_cost += cost
            peak = max(peak, equity)
            max_dd = max(max_dd, 1 - equity / peak)

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

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def run_grid(self) -> pd.DataFrame:
        """Bayesian optimisation over stop loss and take profit."""
        cfg = self.config

        if len(cfg.stop_loss_range) == 1 and len(cfg.take_profit_range) == 1:
            metrics = self._run_single(
                self.df_prepared,
                cfg.stop_loss_range[0],
                cfg.take_profit_range[0],
                self.rng,
            )
            return pd.DataFrame([metrics])

        from skopt import BayesSearchCV
        from skopt.space import Real
        from sklearn.base import BaseEstimator
        from tqdm import tqdm
        from joblib import Parallel, delayed

        class Estimator(BaseEstimator):
            def __init__(self, runner: BacktestRunner, stop_loss: float = 0.02, take_profit: float = 0.04) -> None:
                self.runner = runner
                self.stop_loss = stop_loss
                self.take_profit = take_profit

            def fit(self, X, y=None):  # noqa: D401 - sklearn API
                return self

            def score(self, X, y=None):  # noqa: D401 - sklearn API
                res = self.runner._run_single(self.runner.df_prepared, self.stop_loss, self.take_profit, self.runner.rng)
                return res["sharpe"]

        search_spaces = {
            "stop_loss": Real(min(cfg.stop_loss_range), max(cfg.stop_loss_range)),
            "take_profit": Real(min(cfg.take_profit_range), max(cfg.take_profit_range)),
        }
        est = Estimator(self)
        opt = BayesSearchCV(est, search_spaces, n_iter=10, n_jobs=-1, cv=[(slice(None), slice(None))])
        dummy_X = np.zeros((1, 1))
        with tqdm(total=opt.n_iter, desc="optimising") as bar:
            opt.fit(dummy_X, [0], callback=lambda res: bar.update())

        metrics = Parallel(n_jobs=-1)(
            delayed(self._run_single)(self.df_prepared, p["stop_loss"], p["take_profit"], self.rng)
            for p in opt.cv_results_["params"]
        )
        df = pd.DataFrame(metrics)
        return df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    def run_walk_forward(self, rolling: bool = True) -> pd.DataFrame:
        """Perform walk-forward optimisation."""
        cfg = self.config
        sl_range = list(cfg.stop_loss_range)
        tp_range = list(cfg.take_profit_range)
        df = self.df_prepared
        results: List[dict] = []
        start = 0
        step = 1 if rolling else cfg.window
        while start + cfg.window * 2 <= len(df):
            train = df.iloc[start : start + cfg.window]
            test = df.iloc[start + cfg.window : start + cfg.window * 2]
            regime, _ = classify_regime(train)
            best_sl = sl_range[0]
            best_tp = tp_range[0]
            best_sharpe = -np.inf
            for sl in sl_range:
                for tp in tp_range:
                    metrics = self._run_single(train, sl, tp, self.rng)
                    if metrics["sharpe"] > best_sharpe:
                        best_sharpe = metrics["sharpe"]
                        best_sl = sl
                        best_tp = tp
            test_metrics = self._run_single(test, best_sl, best_tp, self.rng)
            test_metrics.update(
                {
                    "regime": regime,
                    "train_stop_loss_pct": best_sl,
                    "train_take_profit_pct": best_tp,
                }
            )
            results.append(test_metrics)
            start += step

        return pd.DataFrame(results)

    def report(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Return summary statistics and equity curve."""
        import matplotlib.pyplot as plt

        pnl = df_results["pnl"].sum()
        returns = df_results["pnl"].values
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) != 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns))
        sortino = 0.0
        downside = np.std([r for r in returns if r < 0]) or 1e-8
        sortino = np.mean(returns) / downside * np.sqrt(len(returns))
        calmar = 0.0
        dd = df_results["max_drawdown"].max() or 1e-8
        calmar = pnl / dd
        wins = sum(r > 0 for r in returns)
        losses = sum(r <= 0 for r in returns)
        win_loss = wins / max(1, losses)
        expectancy = np.mean(returns) if len(returns) else 0.0

        eq = (1 + df_results["pnl"]).cumprod()
        plt.figure(figsize=(8, 3))
        plt.plot(eq)
        plt.title("Equity Curve")
        plt.xlabel("Trade")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.close()

        return pd.DataFrame(
            {
                "pnl": [pnl],
                "sharpe": [sharpe],
                "sortino": [sortino],
                "calmar": [calmar],
                "win_loss": [win_loss],
                "expectancy": [expectancy],
            }
        )


def backtest(
    symbol: str,
    timeframe: str,
    *,
    since: int,
    limit: int = 1000,
    mode: str = "cex",
    strategy: str | None = None,
    stop_loss_range: Iterable[float] | None = None,
    take_profit_range: Iterable[float] | None = None,
    slippage_pct: float = 0.001,
    fee_pct: float = 0.001,
    misclass_prob: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run regime-aware backtests over parameter ranges."""
    if strategy == "sniper_solana":
        logger.info("Sniper Solana not backtestable \u2013 run paper mode")
        return pd.DataFrame()
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=limit,
        mode=mode,
        stop_loss_range=stop_loss_range,
        take_profit_range=take_profit_range,
        slippage_pct=slippage_pct,
        fee_pct=fee_pct,
        misclass_prob=misclass_prob,
        seed=seed,
    )
    return BacktestRunner(config).run_grid()


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
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=limit,
        mode=mode,
        stop_loss_range=stop_loss_range,
        take_profit_range=take_profit_range,
        slippage_pct=slippage_pct,
        fee_pct=fee_pct,
        misclass_prob=misclass_prob,
        seed=seed,
        window=window,
    )
    return BacktestRunner(config).run_walk_forward()


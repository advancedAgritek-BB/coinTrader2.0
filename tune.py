"""Tune stop-loss and take-profit settings with Optuna.

Usage example
-------------

```
python tune.py --data path/to/ohlcv.csv --symbol XBT/USDT --timeframe 1h
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from crypto_bot.backtest.backtest_runner import BacktestConfig, BacktestRunner


try:
    import bulk_loader
except Exception:  # pragma: no cover - fallback loader
    bulk_loader = None


def load_data(path: Path) -> pd.DataFrame:
    """Return OHLCV data from ``path`` using ``bulk_loader`` if available."""
    if bulk_loader is not None and hasattr(bulk_loader, "load"):
        df = bulk_loader.load(path)
    else:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_json(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


class LocalRunner(BacktestRunner):
    """Backtest runner that uses a preloaded DataFrame."""

    def __init__(self, df: pd.DataFrame, config: BacktestConfig) -> None:
        self._df_source = df
        super().__init__(config)

    def _fetch_data(self) -> pd.DataFrame:  # type: ignore[override]
        return self._df_source


def optimise(df: pd.DataFrame, trials: int, cfg: dict[str, Any]) -> optuna.Study:
    """Run Optuna search over stop-loss and take-profit."""

    df = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=90)]

    def objective(trial: optuna.Trial) -> float:
        sl = trial.suggest_float("stop_loss_pct", 0.005, 0.05)
        tp = trial.suggest_float("take_profit_pct", 0.01, 0.1)
        config = BacktestConfig(
            symbol=cfg.get("symbol", "XBT/USDT"),
            timeframe=cfg.get("timeframe", "1h"),
            since=0,
            limit=len(df),
            stop_loss_range=[sl],
            take_profit_range=[tp],
        )
        runner = LocalRunner(df, config)
        metrics = runner._run_single(runner.df_prepared, sl, tp, runner.rng)
        trial.set_user_attr("metrics", metrics)
        return metrics["sharpe"] or metrics["pnl"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter tuner")
    parser.add_argument("--data", type=Path, required=True, help="OHLCV data file")
    parser.add_argument("--symbol", default="XBT/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    df = load_data(args.data)
    study = optimise(df, args.trials, vars(args))
    best = study.best_trial
    metrics = best.user_attrs.get("metrics", {})
    print("Best parameters:")
    print(best.params)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

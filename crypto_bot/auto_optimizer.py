from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


import yaml
from crypto_bot.utils.symbol_utils import fix_symbol

from crypto_bot.backtest.backtest_runner import BacktestRunner, BacktestConfig
from crypto_bot.utils.logger import LOG_DIR, setup_logger

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
LOG_FILE = LOG_DIR / "optimized_params.json"

logger = setup_logger(__name__, LOG_DIR / "optimizer.log")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    return data


def optimize_strategies() -> Dict[str, Dict[str, float]]:
    """Run backtests for each configured strategy and store best params."""
    cfg = _load_config().get("optimization", {})
    if not cfg.get("enabled"):
        logger.info("Optimization disabled")
        return {}

    param_ranges = cfg.get("parameter_ranges", {})
    bot_cfg = _load_config()
    results: Dict[str, Dict[str, float]] = {}

    for name, ranges in param_ranges.items():
        sl_range: Iterable[float] = ranges.get("stop_loss", [])
        tp_range: Iterable[float] = ranges.get("take_profit", [])

        config = BacktestConfig(
            symbol=bot_cfg.get("symbol", "XBT/USDT"),
            timeframe=bot_cfg.get("timeframe", "1h"),
            since=0,
            limit=1000,
            mode=bot_cfg.get("mode", "cex"),
            stop_loss_range=sl_range,
            take_profit_range=tp_range,
        )

        try:
            df = BacktestRunner(config).run_grid()
        except Exception as exc:  # pragma: no cover - network
            logger.error("Backtest failed for %s: %s", name, exc)
            continue

        df = df.sort_values(["sharpe", "max_drawdown"], ascending=[False, True])
        if df.empty:
            continue
        best = df.iloc[0]
        results[name] = {
            "stop_loss_pct": float(best["stop_loss_pct"]),
            "take_profit_pct": float(best["take_profit_pct"]),
            "sharpe": float(best["sharpe"]),
            "max_drawdown": float(best["max_drawdown"]),
        }
        logger.info(
            "Best for %s sl %.4f tp %.4f sharpe %.2f dd %.2f",
            name,
            results[name]["stop_loss_pct"],
            results[name]["take_profit_pct"],
            results[name]["sharpe"],
            results[name]["max_drawdown"],
        )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(results))
    logger.info("Wrote optimized params to %s", LOG_FILE)
    return results

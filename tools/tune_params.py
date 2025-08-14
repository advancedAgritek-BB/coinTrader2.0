"""Parameter tuning script for selected strategies.

This utility uses scikit-optimize to search over a handful of high
impact parameters for several strategies.  For each candidate set of
parameters a ``tick_sim/mini`` backtest is executed and the average net
PnL after fees across time-split folds is used as the optimisation
objective.

The best set of parameters for each strategy is written to
``config/strategies/<strategy>.yaml`` where it can be used as an override
for future runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from skopt import gp_minimize
from skopt.space import Integer, Real

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Dict] = {
    "Breakout": {
        "space": [
            Integer(20, 80, name="donchian_len"),
            Integer(5, 35, name="bbw_pct_max"),
            Real(0.8, 2.0, name="atr_mult_stop"),
            Real(0.8, 2.0, name="atr_mult_tp"),
        ],
        "config": Path("config/strategies/breakout_bot.yaml"),
    },
    "MeanRevert": {
        "space": [
            Integer(20, 120, name="ema_len"),
            Real(1.0, 3.0, name="z_entry"),
            Real(0.2, 1.5, name="z_exit"),
        ],
        "config": Path("config/strategies/mean_revert_bot.yaml"),
    },
    "Maker": {
        "space": [
            Integer(1, 6, name="edge_margin_bp"),
            Integer(4, 12, name="max_spread_bp"),
        ],
        "config": Path("config/strategies/maker_bot.yaml"),
    },
}

# ---------------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------------

def run_backtest(strategy: str, params: Dict[str, float], start: datetime, end: datetime) -> float:
    """Run ``tick_sim/mini`` backtest and return net PnL after fees.

    The function expects ``tick_sim/mini`` to accept command line
    arguments of the form ``--param value`` and to print a JSON object on
    stdout containing ``net_pnl`` or ``net_pnl_after_fees``.  Any
    failure results in a score of ``0.0``.
    """

    cmd: List[str] = [
        "tick_sim/mini",
        "backtest",
        "--strategy",
        strategy,
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
    ]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.error("Backtest command failed: %s", exc)
        return 0.0

    try:
        # try parsing the last non-empty line as JSON
        line = proc.stdout.strip().splitlines()[-1]
        result = json.loads(line)
    except Exception:  # pragma: no cover - parse safeguard
        logger.error("Could not parse backtest output: %s", proc.stdout)
        return 0.0

    pnl = result.get("net_pnl_after_fees") or result.get("net_pnl")
    try:
        return float(pnl)
    except Exception:  # pragma: no cover - value safeguard
        logger.error("Backtest did not provide PnL: %s", result)
        return 0.0


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def build_folds(weeks: int) -> List[Tuple[datetime, datetime]]:
    """Create non-overlapping weekly time folds ending at ``now``."""
    end = datetime.utcnow()
    start = end - timedelta(days=7 * weeks)
    folds = []
    for i in range(weeks):
        fold_start = start + timedelta(days=7 * i)
        fold_end = fold_start + timedelta(days=7)
        folds.append((fold_start, fold_end))
    return folds


def tune_strategy(name: str, calls: int, weeks: int) -> None:
    """Tune parameters for ``name`` strategy and persist results."""
    spec = STRATEGIES[name]
    space = spec["space"]
    param_names = [dim.name for dim in space]
    folds = build_folds(weeks)

    def objective(values: Iterable[float]) -> float:
        params = {k: v for k, v in zip(param_names, values)}
        total = 0.0
        for start, end in folds:
            total += run_backtest(name, params, start, end)
        # negative because gp_minimize performs minimisation
        return -(total / len(folds))

    result = gp_minimize(objective, space, n_calls=calls, random_state=0, verbose=False)
    best_params = {k: v for k, v in zip(param_names, result.x)}
    score = -result.fun

    print(f"{name} best score: {score:.2f}")
    print(best_params)

    save_path = spec["config"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict = {}
    if save_path.exists():
        with open(save_path) as fh:
            existing = yaml.safe_load(fh) or {}
    existing.update(best_params)
    with open(save_path, "w") as fh:
        yaml.safe_dump(existing, fh)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter tuner")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Strategy to optimise",
    )
    parser.add_argument("--calls", type=int, default=25, help="Number of optimisation calls")
    parser.add_argument("--weeks", type=int, default=3, help="Number of non-overlapping weeks")
    args = parser.parse_args()

    if args.strategy == "all":
        for name in STRATEGIES:
            tune_strategy(name, args.calls, args.weeks)
    else:
        tune_strategy(args.strategy, args.calls, args.weeks)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

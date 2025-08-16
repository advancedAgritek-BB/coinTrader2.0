import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from .logger import LOG_DIR


LOG_FILE = LOG_DIR / "strategy_pnl.csv"
PERFORMANCE_FILE = LOG_DIR / "strategy_performance.json"


def log_pnl(
    regime: str,
    strategy: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    confidence: float,
    direction: str,
) -> None:
    """Append realized PnL information to CSV and JSON logs."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "confidence": confidence,
        "direction": direction,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", header=header, index=False)

    # Append simplified record to strategy_performance.json
    perf_rec = {"timestamp": record["timestamp"], "pnl": float(pnl)}
    try:
        data = (
            json.loads(PERFORMANCE_FILE.read_text())
            if PERFORMANCE_FILE.exists()
            else {}
        )
    except Exception:
        data = {}
    trades = data.setdefault(regime, {}).setdefault(strategy, [])
    trades.append(perf_rec)
    PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))

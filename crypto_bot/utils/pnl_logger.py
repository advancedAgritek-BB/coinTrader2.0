import pandas as pd
from datetime import datetime

from .logger import LOG_DIR
from crypto_bot.selector import bandit


LOG_FILE = LOG_DIR / "strategy_pnl.csv"


def log_pnl(
    strategy: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    confidence: float,
    direction: str,
) -> None:
    """Append realized PnL information to CSV."""
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

    # Update bandit with trade outcome so strategy selection learns
    # across runs.  A positive PnL is treated as a win.
    try:  # pragma: no cover - defensive in case bandit is disabled
        bandit.update(symbol, strategy, pnl > 0)
    except Exception:  # noqa: BLE001
        pass

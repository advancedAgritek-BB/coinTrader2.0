from __future__ import annotations

import json
import time

from .logger import LOG_DIR


def check_and_update(regime: str, tf_seconds: int, intervals: int) -> str:
    """Return commit-locked regime and update lock file.

    Parameters
    ----------
    regime : str
        Current detected regime.
    tf_seconds : int
        Base timeframe in seconds.
    intervals : int
        Number of intervals to lock the regime for.
    """
    file = LOG_DIR / "last_regime.json"
    file.parent.mkdir(parents=True, exist_ok=True)

    if intervals > 0 and file.exists():
        try:
            data = json.loads(file.read_text())
            last_reg = data.get("regime")
            last_ts = float(data.get("timestamp", 0))
            if last_reg and time.time() - last_ts < tf_seconds * intervals and regime != last_reg:
                return last_reg
        except Exception:
            pass

    file.write_text(json.dumps({"regime": regime, "timestamp": time.time()}))
    return regime

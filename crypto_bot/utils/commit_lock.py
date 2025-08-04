from __future__ import annotations

import json
import time

from .logger import LOG_DIR

# Optional Redis client for commit lock persistence
REDIS_CLIENT = None
REDIS_KEY = "commit_lock:last_regime"


def check_and_update(regime: str, tf_seconds: int, intervals: int) -> str:
    """Return commit-locked regime and update persistence layer.

    Parameters
    ----------
    regime : str
        Current detected regime.
    tf_seconds : int
        Base timeframe in seconds.
    intervals : int
        Number of intervals to lock the regime for.
    """

    client = REDIS_CLIENT
    if client is not None:  # pragma: no cover - exercised via tests
        try:
            raw = client.get(REDIS_KEY)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                last_reg = data.get("regime")
                last_ts = float(data.get("timestamp", 0))
                if (
                    intervals > 0
                    and last_reg
                    and time.time() - last_ts < tf_seconds * intervals
                    and regime != last_reg
                ):
                    return last_reg
        except Exception:
            pass
        try:
            client.set(REDIS_KEY, json.dumps({"regime": regime, "timestamp": time.time()}))
        except Exception:
            pass
        return regime

    # Fallback to filesystem persistence if no Redis client is provided
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

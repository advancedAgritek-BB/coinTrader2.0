from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import redis

from .logger import LOG_DIR

REDIS_CLIENT: Optional[redis.Redis] = None
REDIS_KEY = "commit_lock:last_regime"


def check_and_update(
    regime: str,
    tf_seconds: int,
    intervals: int,
    *,
    redis_client: Optional[redis.Redis] = None,
) -> str:
    """Return a commit-locked regime value.

    The last regime is stored either in a Redis instance or on disk. If a
    previous regime exists and is still within the lock window, that value is
    returned; otherwise the provided ``regime`` is persisted and returned.
    """

    client = redis_client or REDIS_CLIENT
    if client is not None:  # pragma: no cover - optional redis path
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

    # Fallback to filesystem persistence
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

    try:
        file.write_text(json.dumps({"regime": regime, "timestamp": time.time()}))
    except Exception:
        pass
    return regime

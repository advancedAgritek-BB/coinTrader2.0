from __future__ import annotations

import json
import time
from typing import Optional

import redis

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

    The last regime is stored in a Redis instance. If a previous regime exists
    and is still within the lock window, that value is returned; otherwise the
    provided ``regime`` is persisted and returned.

    Parameters
    ----------
    regime:
        Proposed regime identifier.
    tf_seconds:
        Timeframe in seconds for each interval.
    intervals:
        Number of intervals the regime should be locked.
    redis_client:
        Optional Redis client. If omitted, the global ``REDIS_CLIENT`` is used.
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

from __future__ import annotations

import json
import time

import redis


def check_and_update(
    regime: str,
    tf_seconds: int,
    intervals: int,
    *,
    redis_client: redis.Redis | None = None,
) -> str:
    """Return commit-locked regime using Redis for coordination.

    Parameters
    ----------
    regime : str
        Current detected regime.
    tf_seconds : int
        Base timeframe in seconds.
    intervals : int
        Number of intervals to lock the regime for.
    redis_client : redis.Redis | None
        Optional Redis client. If not provided, a new client is created.
    """

    if intervals <= 0:
        return regime

    client = redis_client or redis.Redis(decode_responses=True)
    data_key = "commit_lock:last_regime"
    lock_key = f"{data_key}:lock"
    ttl = tf_seconds * intervals

    with client.lock(lock_key, blocking_timeout=1):
        raw = client.get(data_key)
        if raw:
            try:
                payload = json.loads(raw)
                last_reg = payload.get("regime")
                last_ts = float(payload.get("timestamp", 0))
            except Exception:
                last_reg = None
                last_ts = 0.0
            if last_reg and regime != last_reg and time.time() - last_ts < ttl:
                return last_reg

        client.set(
            data_key,
            json.dumps({"regime": regime, "timestamp": time.time()}),
            ex=ttl,
        )

    return regime


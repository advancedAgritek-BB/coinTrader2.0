from __future__ import annotations

from typing import Any, Mapping


def short_selling_enabled(config: Mapping[str, Any]) -> bool:
    """Return True if short selling is enabled in config.

    Supports legacy keys ``allow_short`` and ``allow_shorting`` for backward
    compatibility.  The new canonical location is ``trading.short_selling``.
    """

    trading_cfg = {}
    if isinstance(config, Mapping):
        trading_cfg = config.get("trading") or {}

    if isinstance(trading_cfg, Mapping):
        if "short_selling" in trading_cfg:
            return bool(trading_cfg["short_selling"])
        if "allow_shorting" in trading_cfg:
            return bool(trading_cfg["allow_shorting"])

    if isinstance(config, Mapping) and "allow_short" in config:
        return bool(config["allow_short"])

    return False

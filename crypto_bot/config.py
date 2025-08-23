"""Configuration schema for HFT defaults.

This module defines a simple dataclass-based schema that captures the
high‑frequency trading defaults used throughout the bot.  It provides
per‑timeframe mappings for warmup candles and backfill days so the code can
look up limits for any supported timeframe (e.g., ``1m`` or ``5m``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OHLCVConfig:
    """Settings for persisting and bootstrapping OHLCV data."""

    storage_path: str = "crypto_bot/data/ohlcv"  # Directory for cached candles
    tail_overlap_bars: int = 3  # Bars to re-fetch on update to avoid gaps
    max_bootstrap_bars: int = 1000  # Cap initial load per symbol


@dataclass
class Config:
    """Typed representation of the core runtime configuration."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m"])
    warmup_candles: Dict[str, int] = field(
        default_factory=lambda: {"1m": 2000, "5m": 2000, "15m": 2000, "1h": 1500}
    )
    deep_backfill_days: Dict[str, int] = field(default_factory=dict)
    backfill_days: Dict[str, int] = field(
        default_factory=lambda: {"1m": 2, "5m": 3}
    )
    allowed_quotes: List[str] = field(
        default_factory=lambda: ["USD", "USDT", "USDC", "EUR"]
    )
    ohlcv_chunk_size: int = 20
    ohlcv: OHLCVConfig = field(default_factory=OHLCVConfig)
    hft: bool = True
    strict_cex: bool = False
    denylist_symbols: List[str] = field(default_factory=list)
    require_sentiment: bool = True


# Default global configuration used by modules that expect a ``cfg`` object.
#
# Individual applications may override or replace this instance at runtime,
# but providing it here keeps optional modules decoupled from the loader while
# still allowing them to reference configuration values such as
# ``denylist_symbols`` or ``strict_cex``.
cfg = Config()


def short_selling_enabled(cfg_dict: dict, default: bool = False) -> bool:
    """Return True if short selling is enabled in the configuration.

    This centralises backward-compatibility handling for legacy keys. New
    configurations should define ``trading.short_selling``. Older configs may
    use ``allow_short`` or ``allow_shorting`` either at the top level or within
    ``trading``. This helper resolves the setting with the following
    precedence:

    1. ``trading.short_selling``
    2. ``trading.allow_shorting``
    3. ``allow_short`` (top level)
    4. ``allow_shorting`` (top level)

    Parameters
    ----------
    cfg_dict:
        Configuration dictionary to inspect.
    default:
        Value to return if none of the known keys are present.
    """

    trading = cfg_dict.get("trading", {}) or {}
    if "short_selling" in trading:
        return bool(trading["short_selling"])
    if "allow_shorting" in trading:
        return bool(trading["allow_shorting"])
    if "allow_short" in cfg_dict:
        return bool(cfg_dict["allow_short"])
    if "allow_shorting" in cfg_dict:
        return bool(cfg_dict["allow_shorting"])
    return bool(default)



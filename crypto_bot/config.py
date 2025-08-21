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
        default_factory=lambda: {"1m": 1000, "5m": 600, "15m": 500, "1h": 500}
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


# Default global configuration used by modules that expect a ``cfg`` object.
#
# Individual applications may override or replace this instance at runtime,
# but providing it here keeps optional modules decoupled from the loader while
# still allowing them to reference configuration values such as
# ``denylist_symbols`` or ``strict_cex``.
cfg = Config()



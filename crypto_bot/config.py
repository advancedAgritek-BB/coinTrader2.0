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
class Config:
    """Typed representation of the core runtime configuration."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m"])
    warmup_candles: Dict[str, int] = field(
        default_factory=lambda: {"1m": 1000, "5m": 600}
    )
    backfill_days: Dict[str, int] = field(
        default_factory=lambda: {"1m": 2, "5m": 3}
    )
    allowed_quotes: List[str] = field(
        default_factory=lambda: ["USD", "USDT", "USDC", "EUR"]
    )
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



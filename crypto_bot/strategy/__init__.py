"""Convenience imports for strategy modules."""

from . import (
    bounce_scalper,
    breakout_bot,
    dex_scalper,
    grid_bot,
    mean_bot,
    micro_scalp_bot,
    sniper_bot,
    trend_bot,
)

# Export Solana sniper strategy under a unified name
from crypto_bot.solana.sniper_solana import generate_signal as sniper_solana

__all__ = [
    "bounce_scalper",
    "breakout_bot",
    "dex_scalper",
    "grid_bot",
    "mean_bot",
    "micro_scalp_bot",
    "sniper_bot",
    "trend_bot",
    "sniper_solana",
]


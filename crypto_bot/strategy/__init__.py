"""Convenience imports for strategy modules."""

from . import (
    bounce_scalper,
    dca_bot,
    breakout_bot,
    dex_scalper,
    grid_bot,
    mean_bot,
    micro_scalp_bot,
    sniper_bot,
    trend_bot,
)

# Export Solana sniper strategy module under a unified name
import importlib

# Import the module so callers can access ``sniper_solana.generate_signal`` just
# like before. This keeps backwards compatibility after removing the local
# implementation.
sniper_solana = importlib.import_module("crypto_bot.solana.sniper_solana")

__all__ = [
    "bounce_scalper",
    "breakout_bot",
    "dex_scalper",
    "dca_bot",
    "grid_bot",
    "mean_bot",
    "micro_scalp_bot",
    "sniper_bot",
    "trend_bot",
    "sniper_solana",
]


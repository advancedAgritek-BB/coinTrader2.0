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
    arbitrage_bot,
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
    "grid_bot",
    "mean_bot",
    "micro_scalp_bot",
    "sniper_bot",
    "trend_bot",
    "arbitrage_bot",
    "sniper_solana",
    "high_freq_strategies",
]

# Strategies geared toward lower latency execution. These are prioritized
# by :mod:`crypto_bot.main` when running in HFT mode.
high_freq_strategies = [
    arbitrage_bot.generate_signal,
    micro_scalp_bot.generate_signal,
    dex_scalper.generate_signal,
    bounce_scalper.generate_signal,
]


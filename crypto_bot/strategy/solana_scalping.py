"""Wrapper to expose Solana scalping helpers in the strategy namespace."""

from crypto_bot.solana.scalping import generate_signal

NAME = "solana_scalping"

__all__ = ["generate_signal"]

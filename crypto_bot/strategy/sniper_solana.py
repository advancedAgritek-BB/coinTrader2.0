"""Compatibility wrapper for Solana sniper strategy.

This module exposes the Solana sniping helpers from
``crypto_bot.solana.sniper_solana`` inside the ``crypto_bot.strategy``
namespace so they can be discovered and imported by the strategy loader and
router.
"""

from crypto_bot.solana.sniper_solana import generate_signal, regime_filter

try:  # pragma: no cover - best effort re-export
    from crypto_bot.strategies.sniper_solana import on_trade_filled
except Exception:  # noqa: BLE001 - fallback stub for tests
    async def on_trade_filled(*args, **kwargs):  # type: ignore
        """Fallback no-op when the full implementation is unavailable."""
        return {}

__all__ = ["generate_signal", "on_trade_filled", "regime_filter"]

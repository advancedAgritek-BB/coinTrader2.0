"""Convenience imports for strategy modules."""

from __future__ import annotations

import importlib


def _optional_import(name: str):
    """Import ``name`` from this package, returning ``None`` on failure."""

    try:  # pragma: no cover - optional dependencies
        return importlib.import_module(f".{name}", __name__)
    except Exception:  # pragma: no cover - ignore any import errors
        return None


bounce_scalper = _optional_import("bounce_scalper")
dca_bot = _optional_import("dca_bot")
breakout_bot = _optional_import("breakout_bot")
breakout = _optional_import("breakout")
dex_scalper = _optional_import("dex_scalper")
grid_bot = _optional_import("grid_bot")
mean_bot = _optional_import("mean_bot")
stat_arb_bot = _optional_import("stat_arb_bot")
micro_scalp_bot = _optional_import("micro_scalp_bot")
momentum_bot = _optional_import("momentum_bot")
lstm_bot = _optional_import("lstm_bot")
sniper_bot = _optional_import("sniper_bot")
trend_bot = _optional_import("trend_bot")
dip_hunter = _optional_import("dip_hunter")
meme_wave_bot = _optional_import("meme_wave_bot")
cross_chain_arb_bot = _optional_import("cross_chain_arb_bot")
flash_crash_bot = _optional_import("flash_crash_bot")
range_arb_bot = _optional_import("range_arb_bot")

# Export Solana sniper strategy modules under a unified name
try:  # pragma: no cover - optional dependency
    sniper_solana = importlib.import_module("crypto_bot.strategies.sniper_solana")
except Exception:  # pragma: no cover
    class _Stub:
        @staticmethod
        def generate_signal(*_a, **_k):
            return 0.0, "none"

    sniper_solana = _Stub()
try:  # pragma: no cover - optional dependency
    solana_scalping = importlib.import_module("crypto_bot.solana.scalping")
except Exception:  # pragma: no cover
    solana_scalping = None

__all__ = [
    name
    for name in [
        "bounce_scalper",
        "breakout_bot",
        "breakout",
        "dex_scalper",
        "dca_bot",
        "grid_bot",
        "mean_bot",
        "stat_arb_bot",
        "dip_hunter",
        "micro_scalp_bot",
        "momentum_bot",
        "lstm_bot",
        "meme_wave_bot",
        "cross_chain_arb_bot",
        "flash_crash_bot",
        "range_arb_bot",
        "sniper_bot",
        "trend_bot",
        "sniper_solana",
        "solana_scalping",
    ]
    if globals().get(name) is not None
]


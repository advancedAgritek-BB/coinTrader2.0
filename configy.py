"""Convenience configuration loader for CoinTrader2.0.

This module exposes the configuration loading helpers from
:mod:`crypto_bot.main` so that external scripts can simply ``import
configy`` rather than reaching into the package internals.
"""

from __future__ import annotations

from crypto_bot.main import (
    CONFIG_PATH,
    load_config,
    load_config_async,
    reload_config,
)

__all__ = [
    "CONFIG_PATH",
    "load_config",
    "load_config_async",
    "reload_config",
]


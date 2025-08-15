"""Utilities package initializer.

This module is intentionally kept free of side-effects. Importing the
``crypto_bot.utils`` package should not trigger the import of any heavy
submodules. Consumers should import the required utilities directly from
their respective modules, e.g. ``from crypto_bot.utils import market_loader``
or ``from crypto_bot.utils.market_loader import timeframe_seconds``.
"""

# NOTE: keep this package import side-effect free.
# Do NOT import submodules here.

__all__: list[str] = []


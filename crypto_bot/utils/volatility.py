"""Compatibility layer for volatility helpers.

This module re-exports functions from :mod:`crypto_bot.volatility` so existing
imports continue to work after the helpers were centralized.
"""

from __future__ import annotations

import sys

from crypto_bot import volatility as _vol

# Re-export the base module so attribute patches affect the shared implementation.
sys.modules[__name__] = _vol


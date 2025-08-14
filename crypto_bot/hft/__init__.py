"""Simple HFT engine stubs used for routing tests.

This module provides a lightweight :class:`HFTEngine` and a placeholder
`maker_spread` strategy so the main trading loop can attach HFT strategies
without pulling in heavy dependencies.  The implementation is intentionally
minimal – the engine simply records attached strategies for inspection in
unit tests.
"""

from .engine import HFTEngine
from .strategy import maker_spread

__all__ = ["HFTEngine", "maker_spread"]

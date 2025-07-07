"""Limit the number of simultaneously open trades."""

from __future__ import annotations

from typing import Mapping, Sequence


class OpenPositionGuard:
    """Simple utility enforcing ``max_open_trades``."""

    def __init__(self, max_open_trades: int) -> None:
        self.max_open_trades = max(1, int(max_open_trades))

    def can_open(self, positions: Mapping | Sequence) -> bool:
        """Return ``True`` if another trade may be opened."""
        return len(positions) < self.max_open_trades

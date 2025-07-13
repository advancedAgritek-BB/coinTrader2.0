from __future__ import annotations

from typing import Dict

_last_fill_bar: Dict[str, int] = {}
_active_legs: Dict[str, int] = {}
_current_bar: Dict[str, int] = {}


def update_bar(symbol: str, bar: int) -> None:
    """Record the current bar index for ``symbol``."""
    _current_bar[symbol] = bar


def record_fill(symbol: str) -> None:
    """Record a filled order for ``symbol``."""
    _last_fill_bar[symbol] = _current_bar.get(symbol, 0)
    _active_legs[symbol] = _active_legs.get(symbol, 0) + 1


def in_cooldown(symbol: str, bars: int) -> bool:
    """Return ``True`` if ``symbol`` is within ``bars`` of the last fill."""
    cur = _current_bar.get(symbol)
    last = _last_fill_bar.get(symbol)
    return cur is not None and last is not None and (cur - last) < bars


def active_leg_count(symbol: str) -> int:
    """Return number of active legs for ``symbol``."""
    return _active_legs.get(symbol, 0)


def clear() -> None:
    """Reset state (for tests)."""
    _last_fill_bar.clear()
    _active_legs.clear()
    _current_bar.clear()


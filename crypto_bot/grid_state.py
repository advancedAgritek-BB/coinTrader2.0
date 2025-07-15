from __future__ import annotations

from typing import Dict

_last_fill_bar: Dict[str, int] = {}
_active_legs: Dict[str, int] = {}
_current_bar: Dict[str, int] = {}
_grid_spacing: Dict[str, float] = {}
_grid_step: Dict[str, float] = {}
_last_atr: Dict[str, float] = {}


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


def update_spacing(symbol: str, spacing: float, atr: float) -> None:
    """Persist grid spacing and ATR% for ``symbol``."""
    _grid_spacing[symbol] = spacing
    _last_atr[symbol] = atr


def get_spacing(symbol: str) -> float | None:
    """Return stored grid spacing for ``symbol``."""
    return _grid_spacing.get(symbol)


def get_grid_step(symbol: str) -> float | None:
    """Return stored grid step size for ``symbol``."""
    return _grid_step.get(symbol)


def set_grid_step(symbol: str, step: float) -> None:
    """Persist grid step size for ``symbol``."""
    _grid_step[symbol] = step


def get_last_atr(symbol: str) -> float | None:
    """Return most recent ATR% value for ``symbol`` if known."""
    return _last_atr.get(symbol)


def set_last_atr(symbol: str, atr: float) -> None:
    """Persist latest ATR% for ``symbol``."""
    _last_atr[symbol] = atr


def clear() -> None:
    """Reset state (for tests)."""
    _last_fill_bar.clear()
    _active_legs.clear()
    _current_bar.clear()
    _grid_spacing.clear()
    _grid_step.clear()
    _last_atr.clear()


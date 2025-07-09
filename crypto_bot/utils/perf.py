"""Placeholder performance analytics helpers."""

from __future__ import annotations


def edge(strategy: str, symbol: str) -> float:
    """Return estimated edge for ``strategy`` on ``symbol``.

    The default implementation provides a neutral edge of ``1.0``.
    """
    return 1.0

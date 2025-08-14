"""Placeholder HFT strategies."""

from typing import Any


def maker_spread(*_args: Any, **_kwargs: Any) -> None:
    """Dummy maker-spread strategy.

    The real implementation would place resting orders around the midpoint to
    capture the spread.  In tests we merely need a callable, so this function
    does nothing.
    """

    return None

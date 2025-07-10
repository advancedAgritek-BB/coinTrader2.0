from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def cache_by_id(func: Callable[..., R]) -> Callable[..., R]:
    """Cache ``func`` results keyed by ``id`` of the first argument."""
    cache: Dict[int, R] = {}

    @wraps(func)
    def wrapper(obj: T, *args: Any, **kwargs: Any) -> R:
        key = id(obj)
        if key not in cache:
            cache[key] = func(obj, *args, **kwargs)
        return cache[key]

    return wrapper

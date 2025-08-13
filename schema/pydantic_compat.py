"""Compatibility layer for Pydantic v1 and v2."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

try:  # Pydantic v2 available
    from pydantic import field_validator as _v2_field_validator
    from pydantic import model_validator as _v2_model_validator

    def field_validator(*fields: str, mode: str = "after", **kwargs: Any):
        """Proxy to ``pydantic.field_validator`` when running on v2."""
        return _v2_field_validator(*fields, mode=mode, **kwargs)

    def model_validator(mode: str = "after", **kwargs: Any):
        """Proxy to ``pydantic.model_validator`` when running on v2."""
        return _v2_model_validator(mode=mode, **kwargs)

except Exception:  # Pydantic v1 fallback
    from types import SimpleNamespace
    from pydantic import root_validator as _v1_root_validator
    from pydantic import validator as _v1_validator
    import inspect

    def _make_info(values, config=None, field=None):
        return SimpleNamespace(
            data=values, config=config, field_name=getattr(field, "name", None)
        )

    def field_validator(*fields: str, mode: str = "after", **kwargs: Any):
        """Backport of Pydantic v2's ``field_validator`` for v1."""
        pre = mode == "before"
        kwargs.pop("mode", None)
        kwargs.setdefault("allow_reuse", True)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(func)
            expect_info = len(sig.parameters) > 2

            def wrapper(cls, value, values, config, field):
                info = _make_info(values, config, field)
                if expect_info:
                    return func(cls, value, info)
                return func(cls, value)

            return _v1_validator(*fields, pre=pre, **kwargs)(wrapper)

        return decorator

    def model_validator(mode: str = "after", **kwargs: Any):
        """Backport of Pydantic v2's ``model_validator`` for v1."""
        pre = mode == "before"
        kwargs.setdefault("allow_reuse", True)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(func)
            expect_info = len(sig.parameters) > 2

            def wrapper(cls, values, config, **_):
                info = _make_info(values, config)
                if expect_info:
                    return func(cls, values, info)
                return func(cls, values)

            return _v1_root_validator(pre=pre, **kwargs)(wrapper)

        return decorator


__all__ = ["BaseModel", "Field", "field_validator", "model_validator"]


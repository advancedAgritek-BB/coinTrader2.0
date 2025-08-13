"""Compatibility layer for Pydantic v1 and v2."""

from __future__ import annotations

try:  # Pydantic v2
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:  # Pydantic v1 fallback
    from types import SimpleNamespace

    from pydantic import BaseModel, Field, root_validator, validator

    def field_validator(*fields, **kwargs):
        """Shim that maps Pydantic v2's field_validator to v1's validator."""
        mode = kwargs.pop("mode", None)
        pre = mode == "before"

        def decorator(func):
            def wrapper(cls, v, values, config, field):
                info = SimpleNamespace(data=values, field_name=getattr(field, "name", None))
                return func(cls, v, info)

            wrapper.__name__ = func.__name__
            wrapper.__qualname__ = getattr(func, "__qualname__", func.__name__)
            wrapper.__doc__ = func.__doc__
            return validator(*fields, pre=pre, **kwargs)(wrapper)

        return decorator

    def model_validator(*fields, **kwargs):
        """Shim for model_validator using root_validator in v1."""
        mode = kwargs.pop("mode", None)
        pre = mode == "before"

        def decorator(func):
            def wrapper(cls, values):
                info = SimpleNamespace(data=values)
                return func(cls, info)

            wrapper.__name__ = func.__name__
            wrapper.__qualname__ = getattr(func, "__qualname__", func.__name__)
            wrapper.__doc__ = func.__doc__
            return root_validator(pre=pre, **kwargs)(wrapper)

        return decorator

__all__ = ["BaseModel", "Field", "field_validator", "model_validator"]

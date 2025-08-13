from typing import Any, Callable
from pydantic import BaseModel, Field  # common to v1 & v2

# field_validator: v2 -> v1.validator(pre=...)
try:  # Pydantic v2
    from pydantic import field_validator as _v2_field_validator  # type: ignore

    def field_validator(*fields: str, mode: str = "after", **kwargs: Any):
        # pass through to v2
        return _v2_field_validator(*fields, mode=mode, **kwargs)

except Exception:  # Pydantic v1
    from pydantic import validator as _v1_validator  # type: ignore

    class _Info:
        def __init__(self, values, config, field):
            self.data = values
            self.config = config
            self.field_name = getattr(field, "name", None)

    def field_validator(*fields: str, mode: str = "after", **kwargs: Any):
        # map v2's mode -> v1's pre flag
        pre = mode == "before"
        kwargs.pop("mode", None)
        kwargs.setdefault("allow_reuse", True)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(cls, value, values, config, field):
                info = _Info(values, config, field)
                return func(cls, value, info)

            return _v1_validator(*fields, pre=pre, **kwargs)(wrapper)

        return decorator

# Optional: model_validator alias if you ever use it
try:  # v2
    from pydantic import model_validator as _v2_model_validator  # type: ignore

    def model_validator(mode: str = "after"):
        return _v2_model_validator(mode=mode)

except Exception:  # v1
    from pydantic import root_validator as _v1_root_validator  # type: ignore

    class _ModelInfo:
        def __init__(self, data, config):
            self.data = data
            self.config = config

    def model_validator(mode: str = "after"):
        pre = mode == "before"

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(cls, values, config, **_):
                info = _ModelInfo(values, config)
                return func(cls, values, info)

            return _v1_root_validator(pre=pre)(wrapper)

        return decorator

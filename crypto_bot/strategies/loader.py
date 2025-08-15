from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any, Iterable, List, Optional, Type

from loguru import logger


class BaseStrategy:  # lightweight base; compatible with many patterns
    name: str = "unnamed"
    timeframes: list[str] = ["1m"]

    def __init__(self, **_: Any) -> None:
        pass


def _pick_strategy_class(mod: ModuleType) -> Optional[Type[BaseStrategy]]:
    # 1) Preferred: module.Strategy class
    strategy_cls = getattr(mod, "Strategy", None)
    if inspect.isclass(strategy_cls):
        return strategy_cls  # type: ignore[return-value]

    # 2) Any class subclassing BaseStrategy
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj is BaseStrategy:
            continue
        if issubclass(obj, BaseStrategy):
            return obj

    # 3) Adapter: function-exported strategies
    #   - module.register() -> List[BaseStrategy]
    #   - module.build() -> BaseStrategy
    #   - module.create_strategy() -> BaseStrategy
    for fname in ("register", "build", "create_strategy", "strategy"):
        fn = getattr(mod, fname, None)
        if callable(fn):
            strategy = fn()
            if isinstance(strategy, list) and strategy and isinstance(strategy[0], BaseStrategy):
                # we will instantiate via a tiny wrapper class
                return _wrap_instances_as_class(strategy)
            if isinstance(strategy, BaseStrategy):
                return strategy.__class__
    return None


def _wrap_instances_as_class(instances: List[BaseStrategy]) -> Type[BaseStrategy]:
    """
    Provide a class-type handle for a list of instances returned by register().
    We capture the instances inside a factory to keep loader logic uniform.
    """
    class _Registered(BaseStrategy):
        _instances = instances

        def __init__(self, **kwargs: Any) -> None:
            # no-op; instances are prebuilt
            pass

    return _Registered


def iter_strategy_modules(pkg_name: str) -> Iterable[ModuleType]:
    pkg = importlib.import_module(pkg_name)
    if not hasattr(pkg, "__path__"):
        return
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_") or m.name == "loader":
            continue
        try:
            yield importlib.import_module(f"{pkg_name}.{m.name}")
        except Exception as e:
            logger.error(f"Failed to import strategy module {m.name}: {e}")


def load_strategies(pkg_name: str = "crypto_bot.strategies", **kwargs: Any) -> list[BaseStrategy]:
    loaded: list[BaseStrategy] = []

    for mod in iter_strategy_modules(pkg_name):
        cls = _pick_strategy_class(mod)
        if cls is None:
            logger.warning(f"Strategy module {mod.__name__.split('.')[-1]} has no Strategy class; skipping.")
            continue

        try:
            # Instantiate with forgiving kwargs (strategies should accept **kwargs)
            obj = cls(**kwargs)  # type: ignore[call-arg]
            if hasattr(obj, "_instances"):  # registered list case
                loaded.extend(obj._instances)  # type: ignore[attr-defined]
            else:
                loaded.append(obj)
            logger.info(f"Loaded strategy: {getattr(obj, 'name', cls.__name__)}")
        except Exception as e:
            logger.error(f"Failed to instantiate strategy from {mod.__name__}: {e}")

    if not loaded:
        logger.error("No strategies loaded! Trading disabled until strategies import cleanly.")
    return loaded

import importlib
import pkgutil
import traceback
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Modules inside ``crypto_bot.strategy`` that are not actual strategy
# implementations and should be ignored when auto-discovering available
# strategies.
_NON_STRATEGY_MODS = {
    "loader",
    "registry",
    "evaluator",
    "hft_engine",
    "maker_spread",
}


def _default_module_names(package_name: str) -> Set[str]:
    """Return all importable strategy module names under ``package_name``.

    Any modules listed in :data:`_NON_STRATEGY_MODS` are skipped.
    """

    pkg = importlib.import_module(package_name)
    return {
        m.name
        for m in pkgutil.iter_modules(pkg.__path__)
        if m.name not in _NON_STRATEGY_MODS
    }


def _module_name(mod) -> str:
    """Return a best effort name for ``mod``.

    The module's ``__name__`` basename is used when available and truthy;
    otherwise the name is derived from the file path.  As a final fallback,
    ``"unknown"`` is returned.
    """

    base = getattr(mod, "__name__", "")
    base = base.split(".")[-1] if base else ""
    if base:
        return base
    file = getattr(mod, "__file__", "")
    if file:
        stem = Path(file).stem
        if stem:
            return stem
    return "unknown"

class _SimpleRegistry:
    def __init__(self):
        self._items: Dict[str, Any] = {}
    def register(self, name: str, strategy: Any):
        self._items[name] = strategy
    def items(self):
        return self._items.items()

def _as_list(x) -> List[Any]:
    if x is None: return []
    if isinstance(x, dict): return list(x.values())
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def _instantiate(obj: Any) -> Any | None:
    try:
        return obj() if callable(obj) else obj
    except Exception:
        logger.error("Strategy instantiation failed:\n%s", traceback.format_exc())
        return None

def _discover(mod) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # 1) class Strategy
    cls = getattr(mod, "Strategy", None)
    if cls:
        inst = _instantiate(cls)
        if inst:
            name = getattr(inst, "name", None)
            if not name:
                name = _module_name(mod)
            out[name] = inst
    # 2) STRATEGIES / ALL_STRATEGIES / strategies
    for attr in ("STRATEGIES", "__all_strategies__", "strategies", "ALL_STRATEGIES"):
        if hasattr(mod, attr):
            for obj in _as_list(getattr(mod, attr)):
                inst = _instantiate(obj)
                if inst:
                    name = getattr(inst, "name", None)
                    if not name:
                        name = obj.__class__.__name__
                    out[name] = inst
    # 3) get_strategies()
    for attr in ("get_strategies", "strategies_factory"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            try:
                for obj in _as_list(fn()):
                    inst = _instantiate(obj)
                    if inst:
                        name = getattr(inst, "name", None)
                        if not name:
                            name = obj.__class__.__name__
                        out[name] = inst
            except Exception:
                logger.error("get_strategies() failed:\n%s", traceback.format_exc())
    # 4) register(registry)
    reg = getattr(mod, "register", None)
    if callable(reg):
        try:
            r = _SimpleRegistry()
            reg(r)
            for name, obj in r.items():
                inst = _instantiate(obj)
                if inst: out[name] = inst
        except Exception:
            logger.error("register() failed:\n%s", traceback.format_exc())
    # 5) Fallback: plain modules exposing ``generate_signal``
    sig = getattr(mod, "generate_signal", None)
    if callable(sig):
        filt = getattr(mod, "regime_filter", None)

        if filt is None:
            class _DefaultFilter:
                """Regime filter that matches any regime."""

                @staticmethod
                def matches(_regime: str) -> bool:  # pragma: no cover - trivial
                    return True

            filt = _DefaultFilter

        name = getattr(mod, "NAME", None)
        if not name:
            name = _module_name(mod)
        out[name] = SimpleNamespace(
            name=name,
            generate_signal=sig,
            regime_filter=filt,
        )
    return out

def load_strategies(package_name: str = "crypto_bot.strategy",
                    enabled: Optional[Iterable[str]] = None):
    enabled = set(enabled or _default_module_names(package_name))
    loaded, errors = {}, {}
    pkg = importlib.import_module(package_name)

    seen_mods: Set[str] = set()
    for m in pkgutil.iter_modules(pkg.__path__):
        mod_name = m.name
        seen_mods.add(mod_name)
        if mod_name not in enabled:
            logger.debug("Strategy module %s not enabled; skipping.", mod_name)
            continue
        try:
            mod = importlib.import_module(f"{package_name}.{mod_name}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Failed to import module %s: %s\n%s", mod_name, e, tb)
            errors[mod_name] = tb
            continue
        found = _discover(mod)
        if not found:
            logger.warning("No strategies discovered in module %s.", mod_name)
        for name, inst in found.items():
            loaded[name] = inst
            logger.info("Loaded strategy %s from module %s", name, mod_name)

    # Handle explicitly requested modules that aren't discoverable via pkgutil
    for mod_name in enabled - seen_mods:
        try:
            mod = importlib.import_module(f"{package_name}.{mod_name}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Failed to import module %s: %s\n%s", mod_name, e, tb)
            errors[mod_name] = tb
            continue
        found = _discover(mod)
        if not found:
            logger.warning("No strategies discovered in module %s.", mod_name)
        for name, inst in found.items():
            loaded[name] = inst
            logger.info("Loaded strategy %s from module %s", name, mod_name)

    if not loaded:
        logger.error("No strategies loaded! Trading disabled until strategies import cleanly.")
    return loaded, errors

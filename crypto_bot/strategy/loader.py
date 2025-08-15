import importlib, pkgutil, traceback, logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)
# Default strategy modules enabled when ``load_strategies`` is called without
# an explicit ``enabled`` list.  These names must match the actual module names
# within :mod:`crypto_bot.strategy`.
DEFAULT_ENABLED = {"grid_bot", "trend_bot", "micro_scalp_bot", "sniper_solana"}

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
            out[getattr(inst, "name", mod.__name__.split(".")[-1])] = inst
    # 2) STRATEGIES / ALL_STRATEGIES / strategies
    for attr in ("STRATEGIES", "__all_strategies__", "strategies", "ALL_STRATEGIES"):
        if hasattr(mod, attr):
            for obj in _as_list(getattr(mod, attr)):
                inst = _instantiate(obj)
                if inst:
                    out[getattr(inst, "name", obj.__class__.__name__)] = inst
    # 3) get_strategies()
    for attr in ("get_strategies", "strategies_factory"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            try:
                for obj in _as_list(fn()):
                    inst = _instantiate(obj)
                    if inst:
                        out[getattr(inst, "name", obj.__class__.__name__)] = inst
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
    return out

def load_strategies(package_name: str = "crypto_bot.strategy",
                    enabled: Optional[Iterable[str]] = None):
    enabled = set(enabled or DEFAULT_ENABLED)
    loaded, errors = {}, {}
    pkg = importlib.import_module(package_name)

    for m in pkgutil.iter_modules(pkg.__path__):
        mod_name = m.name
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

    if not loaded:
        logger.error("No strategies loaded! Trading disabled until strategies import cleanly.")
    return loaded, errors

import importlib
import logging
import pkgutil
import traceback

logger = logging.getLogger(__name__)


def load_strategies(package_name: str = "crypto_bot.strategy"):
    loaded = {}
    errors = {}

    pkg = importlib.import_module(package_name)
    for m in pkgutil.iter_modules(pkg.__path__):
        name = m.name
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{package_name}.{name}")
            cls = getattr(mod, "Strategy", None)
            if cls is None:
                logger.warning("Strategy module %s has no `Strategy` class; skipping.", name)
                continue
            loaded[name] = cls()
            logger.info("Loaded strategy %s", name)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Failed to import strategy %s: %s\n%s", name, e, tb)
            errors[name] = tb

    if not loaded:
        logger.error("No strategies loaded! Trading disabled until strategies import cleanly.")
    return loaded, errors

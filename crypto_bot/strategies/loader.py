from importlib import import_module
from loguru import logger


def load_strategies(mode: str, names: list[str]) -> list:
    """
    mode: 'cex' | 'onchain' | 'auto'
    names: strategy module names to attempt import
    """
    loaded = []
    seen = set()
    for mod_name in names:
        if mod_name in seen:
            continue
        seen.add(mod_name)

        # Mode gating
        if mod_name == "sniper_solana" and mode == "cex":
            logger.info("Skipping sniper_solana in CEX mode.")
            continue
        if mod_name != "sniper_solana" and mode == "onchain":
            # Only the on-chain sniper is relevant in onchain mode
            continue

        try:
            m = import_module(f"crypto_bot.strategies.{mod_name}")
        except ModuleNotFoundError:
            try:
                m = import_module(f"crypto_bot.strategy.{mod_name}")
            except ModuleNotFoundError:
                logger.error("Strategy %s not found", mod_name)
                continue
            except Exception as e:
                logger.error(f"Failed to import strategy {mod_name}: {e!r}")
                continue
        except Exception as e:
            logger.error(f"Failed to import strategy {mod_name}: {e!r}")
            continue

        Strategy = getattr(m, "Strategy", None)
        if Strategy is None:
            logger.warning(
                f"Strategy module {mod_name} has no `Strategy` class; skipping."
            )
            continue
        loaded.append(Strategy())
        logger.info(f"Loaded strategy: {mod_name}")
    return loaded

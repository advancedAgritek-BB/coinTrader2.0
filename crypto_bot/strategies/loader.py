"""Backward compatible loader that delegates to :mod:`crypto_bot.strategy`."""

from loguru import logger

from crypto_bot.strategy.loader import load_strategies as _load_strategies


def load_strategies(
    mode: str, names: list[str] | None = None, return_errors: bool = False
) -> list | tuple[list, dict]:
    """Return a list of instantiated strategy objects.

    Parameters
    ----------
    mode:
        Execution mode, e.g. ``'cex'`` or ``'onchain'``.  The parameter is kept
        for API compatibility; currently it only gates the Solana sniper when
        running in CEX mode.
    names:
        Optional iterable of module names.  When omitted, all strategies under
        :mod:`crypto_bot.strategy` are loaded.
    """

    loaded, errors = _load_strategies(enabled=names)
    strategies = []
    for name, inst in loaded.items():
        if name == "sniper_solana" and mode == "cex":
            logger.info("Skipping sniper_solana in CEX mode.")
            continue
        if name != "sniper_solana" and mode == "onchain":
            # Only the on-chain sniper is relevant in onchain mode
            continue
        strategies.append(inst)

    for mod_name, err in errors.items():
        logger.error("Failed to load strategy {}: {}", mod_name, err)

    if return_errors:
        return strategies, errors
    return strategies


__all__ = ["load_strategies"]


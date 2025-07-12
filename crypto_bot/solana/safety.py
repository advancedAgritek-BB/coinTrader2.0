"""Safety checks for Solana sniping."""

from __future__ import annotations

from typing import Mapping
import warnings

from .watcher import NewPoolEvent


def is_safe(event: NewPoolEvent, cfg: Mapping[str, object]) -> bool:
    """Return ``True`` when the pool passes basic safety heuristics."""

    if not event.pool_address or not event.token_mint:
        return False

    blocklist_values = list(cfg.get("freeze_blocklist", []))
    if not blocklist_values and cfg.get("freeze_blacklist"):
        warnings.warn(
            "freeze_blacklist is deprecated; use freeze_blocklist",
            DeprecationWarning,
            stacklevel=2,
        )
        blocklist_values = list(cfg.get("freeze_blacklist", []))
    freeze_blocklist = blocklist_values
    if event.freeze_authority and event.freeze_authority in freeze_blocklist:
        return False

    mint_values = list(cfg.get("mint_blocklist", []))
    if not mint_values and cfg.get("mint_blacklist"):
        warnings.warn(
            "mint_blacklist is deprecated; use mint_blocklist",
            DeprecationWarning,
            stacklevel=2,
        )
        mint_values = list(cfg.get("mint_blacklist", []))
    mint_blocklist = mint_values
    if event.mint_authority and event.mint_authority in mint_blocklist:
        return False

    min_liquidity = float(cfg.get("min_liquidity", 0))
    if event.liquidity < min_liquidity:
        return False

    max_dev_share = float(cfg.get("max_dev_share", 100))
    dev_share = float(cfg.get("dev_share", 0))
    if dev_share > max_dev_share:
        return False

    return True

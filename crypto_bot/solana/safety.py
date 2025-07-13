"""Safety checks for Solana sniping."""

from __future__ import annotations

from typing import Mapping

from .watcher import NewPoolEvent


def is_safe(event: NewPoolEvent, cfg: Mapping[str, object]) -> bool:
    """Return ``True`` when the pool passes basic safety heuristics."""

    if not event.pool_address or not event.token_mint:
        return False

    freeze_blacklist = cfg.get("freeze_blacklist", [])
    if event.freeze_authority and event.freeze_authority in freeze_blacklist:
        return False

    mint_blacklist = cfg.get("mint_blacklist", [])
    if event.mint_authority and event.mint_authority in mint_blacklist:
        return False

    min_liquidity = float(cfg.get("min_liquidity", 0))
    if event.liquidity < min_liquidity:
        return False

    max_dev_share = float(cfg.get("max_dev_share", 100))
    dev_share = float(cfg.get("dev_share", 0))
    if dev_share > max_dev_share:
        return False

    return True

"""Solana sniping utilities."""

from .watcher import NewPoolEvent, PoolWatcher
from .meme_wave_runner import start_runner
from .runner import run

try:  # optional imports for lightweight environments
    from .sniper_solana import rug_check, score_new_pool
    from .api_helpers import helius_ws, fetch_jito_bundle
    from .scanner import get_solana_new_tokens
    from .token_utils import get_token_accounts
    from .pyth_utils import get_pyth_price
    from .prices import fetch_solana_prices
except Exception:  # pragma: no cover - allow partial functionality
    rug_check = score_new_pool = lambda *a, **k: None
    helius_ws = fetch_jito_bundle = lambda *a, **k: None
    get_solana_new_tokens = lambda *a, **k: []
    get_token_accounts = lambda *a, **k: []
    get_pyth_price = lambda *a, **k: None
    fetch_solana_prices = lambda *a, **k: {}

__all__ = [
    "NewPoolEvent",
    "PoolWatcher",
    "run",
    "helius_ws",
    "fetch_jito_bundle",
    "get_solana_new_tokens",
    "start_runner",
    "score_new_pool",
    "rug_check",
    "get_token_accounts",
    "get_pyth_price",
    "fetch_solana_prices",
]

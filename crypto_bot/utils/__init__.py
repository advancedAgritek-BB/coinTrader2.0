from .gecko import gecko_request
from .kraken import get_ws_token
from .eval_queue import compute_batches, build_priority_queue
from .market_loader import (
    fetch_ohlcv_async,
    fetch_order_book_async,
    load_kraken_symbols,
    load_ohlcv,
    load_ohlcv_parallel,
    timeframe_seconds,
    update_ohlcv_cache,
)
from .pair_cache import load_liquid_pairs
import crypto_bot.utils.telegram as telegram

try:  # optional dependency with known syntax issues
    from .notifier import Notifier
except Exception:  # pragma: no cover - optional
    Notifier = None
# Symbol filtering utilities import is optional because the module has
# heavy async dependencies and some environments may not need it during
# initialization. Import it lazily where required.
try:  # optional heavy dependency
    from .symbol_utils import get_filtered_symbols, fix_symbol
except Exception:  # pragma: no cover - allow import without symbol_utils
    get_filtered_symbols = None
    fix_symbol = None
from .stats import zscore
from .commit_lock import check_and_update
from .telemetry import telemetry
try:
    from .solana_scanner import get_solana_new_tokens as utils_get_solana_new_tokens
except Exception:  # pragma: no cover - optional dependency
    utils_get_solana_new_tokens = None
from .pyth import get_pyth_price
from .pyth_utils import async_get_pyth_price, get_pyth_price as sync_get_pyth_price
from .token_registry import load_token_mints, TOKEN_MINTS, fetch_from_jupiter
from .constants import NON_SOLANA_BASES
from .ml_utils import is_ml_available, ML_AVAILABLE
from .lunarcrush_client import LunarCrushClient
try:  # optional ML utilities
    from .ml_utils import ML_AVAILABLE
except Exception:  # pragma: no cover - optional dependency
    ML_AVAILABLE = False

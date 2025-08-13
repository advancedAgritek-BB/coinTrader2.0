import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_missing_meta_seen = defaultdict(int)

def log_missing_metadata(symbol: str):
    _missing_meta_seen[symbol] += 1
    # Only log the first occurrence and then every 50th to reduce noise
    if _missing_meta_seen[symbol] == 1 or _missing_meta_seen[symbol] % 50 == 0:
        logger.info("No metadata for %s (seen=%d)", symbol, _missing_meta_seen[symbol])

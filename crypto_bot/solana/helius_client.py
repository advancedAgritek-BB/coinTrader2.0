import logging
import os

logger = logging.getLogger(__name__)

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY") or os.getenv("HELIUS_KEY") or ""
helius_available = bool(HELIUS_API_KEY)

if not helius_available:
    logger.warning("Helius unavailable; on-chain metadata checks skipped.")

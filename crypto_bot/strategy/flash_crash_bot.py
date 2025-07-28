from typing import Optional, Tuple

import pandas as pd

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Dummy flash crash strategy returning no signal.

    The actual logic is not implemented in tests; this stub allows routing
    behaviour to be validated.
    """
    symbol = config.get("symbol") if config else ""
    logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
    return 0.0, "none"

import pandas as pd
from typing import Optional, Tuple


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return signal when a potential flash crash is detected."""
    if df is None or df.empty or "close" not in df:
        return 0.0, "none"

    if len(df) >= 2:
        prev = df["close"].iloc[-2]
        last = df["close"].iloc[-1]
        if last < prev * 0.9:
            return 1.0, "long"
    return 0.0, "none"
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
from typing import Tuple, Optional
import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return long signal on sudden drops with high volume."""
    if df is None or len(df) < 2:
        return 0.0, "none"

    params = config.get("flash_crash", {}) if config else {}
    drop_pct = float(params.get("drop_pct", 0.1))
    vol_mult = float(params.get("volume_mult", 5.0))
    vol_window = int(params.get("vol_window", 20))
    ema_window = int(params.get("ema_window", 200))
    atr_norm = bool(params.get("atr_normalization", True))

    lookback = max(vol_window, ema_window)
    recent = df.iloc[-(lookback + 1) :]

    vol_ma = recent["volume"].rolling(vol_window).mean()
    ema = ta.trend.ema_indicator(recent["close"], window=ema_window)

    last = recent.iloc[-1]
    prev_close = recent["close"].iloc[-2]

    drop = (prev_close - last["close"]) / prev_close if prev_close else 0.0
    vol_ok = (
        pd.notna(vol_ma.iloc[-1])
        and vol_ma.iloc[-1] > 0
        and last["volume"] >= vol_ma.iloc[-1] * vol_mult
    )
    ema_ok = pd.isna(ema.iloc[-1]) or last["close"] < ema.iloc[-1]

    if drop >= drop_pct and vol_ok and ema_ok:
        score = min(drop / drop_pct, 1.0)
        if atr_norm:
            score = normalize_score_by_volatility(df, score)
        score = max(0.0, min(float(score), 1.0))
        return score, "long"

    return 0.0, "none"


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"

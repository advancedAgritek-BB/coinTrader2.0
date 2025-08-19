from typing import Optional, Tuple

import logging
import pandas as pd

logger = logging.getLogger(__name__)
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once
NAME = "lstm_bot"

try:  # pragma: no cover - optional trainer
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer unavailable
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("lstm_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return LSTM-based momentum signal."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    if df is None or df.empty:
        return 0.0, "none"

    params = config or {}
    _ = params.get("model_path")  # preserved for backward compatibility
    seq_len = int(params.get("sequence_length", 50))
    threshold = float(params.get("threshold_pct", 0.0))

    if len(df) < seq_len:
        return 0.0, "none"

    score = 0.0
    if MODEL is not None:
        try:  # pragma: no cover - best effort
            score = float(MODEL.predict(df.tail(seq_len)))
        except Exception:
            score = 0.0

    direction = "none"
    if score > threshold:
        direction = "long"
    elif score < -threshold:
        direction = "short"
    else:
        score = 0.0

    return score, direction


class regime_filter:
    """Match all regimes."""

    @staticmethod
    def matches(_regime: str) -> bool:
        return True

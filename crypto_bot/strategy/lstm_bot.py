import pandas as pd
from typing import Optional, Tuple

try:  # pragma: no cover - optional trainer
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("lstm_bot")
except Exception:  # pragma: no cover - fallback when trainer unavailable
    MODEL = None


def generate_signal(
    df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[float, str]:
    """Return LSTM-based momentum signal."""
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

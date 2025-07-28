import pandas as pd
from typing import Optional, Tuple

try:  # pragma: no cover - optional trainer
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("lstm_bot")
except Exception:  # pragma: no cover - fallback when trainer unavailable
    MODEL = None


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return trading signal using an LSTM price predictor."""
    if df.empty or "close" not in df:
        return 0.0, "none"

    if MODEL is None:
        return 0.0, "none"

    try:  # pragma: no cover - best effort
        pred = MODEL.predict(df)
        if isinstance(pred, (list, tuple)):
            pred_price = float(pred[-1])
        elif hasattr(pred, "item"):
            pred_price = float(pred.item())
        else:
            pred_price = float(pred)
    except Exception:
        return 0.0, "none"

    current = float(df["close"].iloc[-1])
    if current == 0:
        return 0.0, "none"

    score = (pred_price - current) / current
    if score > 0.01:
        return float(score), "long"
    if score < -0.01:
        return float(abs(score)), "short"
    return 0.0, "none"

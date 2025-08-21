"""Gradient boosting fallback model for regime classification."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from .model_data import MODEL_B64
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "bot.log")

_model = None
_scaler = None


def _heuristic_model():
    """Return a minimal heuristic model used when decoding/loading fails."""

    class _SimpleModel:
        def predict_proba(self, X):
            change = float(X[0][0])
            prob = 1.0 if change > 0 else 0.0
            return np.array([[1.0 - prob, prob]])

    return _SimpleModel()


def load_model():
    """Decode and return the embedded LightGBM model."""
    global _model, _scaler
    if _model is None:
        try:
            data = base64.b64decode(MODEL_B64)
            obj = joblib.load(io.BytesIO(data))
            if isinstance(obj, dict):
                _model = obj.get("model")
                _scaler = obj.get("scaler")
            else:
                _model = obj
        except Exception as exc:  # pragma: no cover - log decode/load failure
            logger.error("Failed to decode or load ML model: %s", exc)
            _model = _heuristic_model()
            _scaler = None
    return _model


def predict_regime(df: pd.DataFrame) -> Tuple[str, float]:
    """Predict regime label and confidence using the embedded model."""
    if df is None or len(df) < 2:
        return "unknown", 0.0
    model = load_model()
    scaler = _scaler
    if model is None:
        return "unknown", 0.0
    change = df["close"].iloc[-1] - df["close"].iloc[0]
    X = np.array([[change]])
    if scaler is not None:
        X = scaler.transform(X)
    prob = float(model.predict_proba(X)[0, 1])
    if prob > 0.55:
        label = "trending"
    elif prob < 0.45:
        label = "mean-reverting"
    else:
        label = "sideways"
    confidence = abs(prob - 0.5) * 2
    return label, confidence

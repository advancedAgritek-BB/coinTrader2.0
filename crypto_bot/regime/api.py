"""Regime model API with resilient fallbacks.

This module attempts to load the latest trained regime model from a
Supabase-backed registry.  Should the registry be unavailable or the
model fail to load/execute, a lightweight technical-analysis baseline is
used instead.  The baseline relies solely on ``pandas``/``numpy`` and
therefore works in minimal environments.
This module attempts to load the latest trained regime model from Supabase via
``load_latest_regime``.  Should the remote retrieval or model inference fail, a
lightweight technical-analysis baseline is used instead.  The baseline relies
solely on ``pandas``/``numpy`` and therefore works in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:  # Import lazily guarded; registry may be unavailable in some envs
    from crypto_bot.regime import registry as _registry
except Exception:  # pragma: no cover - registry is optional
    _registry = None
# Regime models are retrieved from Supabase via ``load_latest_regime``.  This
# import is intentionally lightweight; any networking or dependency issues are
# handled within the prediction routine which falls back to the baseline
# implementation on failure.
from crypto_bot.regime.registry import load_latest_regime


Action = Literal["long", "flat", "short"]


@dataclass
class Prediction:
    """Container for regime predictions."""

    action: Action
    score: float
    regime: Optional[str] = None
    meta: dict | None = None


def _baseline_action(features: pd.DataFrame) -> Prediction:
    """Pure-Python fallback prediction.

    Uses ``close`` column if available, otherwise the last numeric column of
    ``features``.  Computes RSI and an EMA crossover to determine a trading
    action.  This approach is intentionally simple but provides deterministic
    behaviour when a trained model or registry is unavailable.
    """

    cols = [c.lower() for c in features.columns]
    if "close" in cols:
        price = features.iloc[:, cols.index("close")]
    else:
        num_cols = features.select_dtypes(include=[np.number]).columns
        price = (
            features[num_cols[-1]]
            if len(num_cols)
            else pd.Series(np.arange(len(features)))
        )

    rsi = _rsi(price).iloc[-1]
    ema_fast = price.ewm(span=8, adjust=False).mean().iloc[-1]
    ema_slow = price.ewm(span=21, adjust=False).mean().iloc[-1]
    cross = float(np.sign(ema_fast - ema_slow))

    if rsi >= 65 and cross >= 0:
        score = float(min(1.0, 0.5 + (rsi - 65) / 35 + 0.25 * cross))
        return Prediction("long", score, meta={"source": "fallback"})
    if rsi <= 35 and cross <= 0:
        score = float(min(1.0, 0.5 + (35 - rsi) / 35 + 0.25 * (-cross)))
        return Prediction("short", score, meta={"source": "fallback"})

    return Prediction("flat", 0.5, meta={"source": "fallback"})


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""

    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _load_model_from_bytes(blob: bytes):
    """Deserialize a model from bytes.

    Joblib is attempted first (if available) and falls back to the standard
    ``pickle`` module.  Importing ``joblib`` lazily avoids mandatory
    dependencies for callers who only rely on the baseline prediction.
    """

    try:
        import joblib  # type: ignore

        return joblib.load(BytesIO(blob))
    except Exception:  # pragma: no cover - joblib may be missing or fail
        import pickle

        return pickle.loads(blob)


def predict(features: pd.DataFrame, symbol: str = "BTCUSDT") -> Prediction:
    """Predict the trading regime for the provided ``features``.

    The function tries to fetch the latest trained model and its accompanying
    metadata from Supabase via :func:`load_latest_regime`.  If any part of the
    retrieval or model execution fails, a deterministic baseline action is
    returned instead.
    """

    if _registry is None:  # Registry not imported or unavailable
        return _baseline_action(features)

    try:
        model, meta = _registry.load_latest_regime(symbol)
    try:
        # ``load_latest_regime`` returns the raw model bytes and metadata that
        # includes the feature list used during training.
        blob, meta = load_latest_regime(symbol)
        feat_list = meta.get("feature_list")
        features_df = features
        if feat_list:
            available = [c for c in feat_list if c in features.columns]
            if available:
                features_df = features[available]

        proba = model.predict_proba(features.tail(1))  # type: ignore[attr-defined]
        model = _load_model_from_bytes(blob)
        proba = model.predict_proba(features_df.tail(1))  # type: ignore[attr-defined]
        proba = getattr(proba, "ravel", lambda: proba)()
        idx = int(np.argmax(proba))
        label_order = meta.get("label_order", [-1, 0, 1])
        mapping = {-1: "short", 0: "flat", 1: "long"}
        class_id = label_order[idx] if idx < len(label_order) else 0
        action = mapping.get(class_id, "flat")
        score = (
            float(proba[idx]) if hasattr(proba, "__len__") else float(proba)
        )
        return Prediction(action=action, score=score, meta=meta)
    except Exception:  # pragma: no cover - registry or model issues
        score = float(proba[idx]) if hasattr(proba, "__len__") else float(proba)
        return Prediction(action=action, score=score, meta=meta)
    except Exception:  # pragma: no cover - network or model failure
        return _baseline_action(features)


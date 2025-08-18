"""Regime model API with resilient fallbacks.

This module attempts to load the latest trained regime model from a
Supabase-backed registry.  If credentials are missing or the remote
retrieval/model execution fails, a lightweight technical-analysis
baseline is used instead.  The baseline relies solely on ``pandas`` and
``numpy`` and therefore works in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Optional
import logging
import os

import numpy as np
import pandas as pd

from crypto_bot.regime.registry import load_latest_regime


logger = logging.getLogger(__name__)


Action = Literal["long", "flat", "short"]


@dataclass
class Prediction:
    """Container for regime predictions."""

    action: Action
    score: float
    regime: Optional[str] = None
    meta: dict | None = None


def _baseline_action(features: pd.DataFrame) -> Prediction:
    """Pure-Python fallback prediction."""

    cols = [c.lower() for c in features.columns]
    if "close" in cols:
        price = features.iloc[:, cols.index("close")]
    else:
        num_cols = features.select_dtypes(include=[np.number]).columns
        price = features[num_cols[-1]] if len(num_cols) else pd.Series(
            np.arange(len(features))
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
    """Deserialize a model from bytes."""

    try:
        import joblib  # type: ignore

        return joblib.load(BytesIO(blob))
    except Exception:  # pragma: no cover - joblib may be missing or fail
        import pickle

        return pickle.loads(blob)


def _have_supabase_creds() -> bool:
    """Return ``True`` if Supabase credentials are present."""

    url_ok = bool(os.getenv("SUPABASE_URL"))
    key_ok = any(
        os.getenv(k)
        for k in [
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_KEY",
            "SUPABASE_API_KEY",
            "SUPABASE_ANON_KEY",
        ]
    )
    return url_ok and key_ok


def predict(features: pd.DataFrame, symbol: str = "BTCUSDT") -> Prediction:
    """Predict the trading regime for the provided ``features``."""

    if not _have_supabase_creds():
        logger.warning(
            "Supabase credentials missing; set SUPABASE_URL and one of "
            "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_KEY or SUPABASE_API_KEY. "
            "Falling back to heuristic regime (set features.ml=false to run heuristics)."
        )
        return _baseline_action(features)

    try:
        blob, meta = load_latest_regime(symbol)
        if not isinstance(blob, (bytes, bytearray)):
            return _baseline_action(features)
        feat_list = meta.get("feature_list")
        features_df = features
        if feat_list:
            available = [c for c in feat_list if c in features.columns]
            if available:
                features_df = features[available]

        model = _load_model_from_bytes(blob)
        proba = model.predict_proba(features_df.tail(1))  # type: ignore[attr-defined]
        proba = getattr(proba, "ravel", lambda: proba)()
        idx = int(np.argmax(proba))
        label_order = meta.get("label_order", [-1, 0, 1])
        mapping = {-1: "short", 0: "flat", 1: "long"}
        class_id = label_order[idx] if idx < len(label_order) else 0
        action = mapping.get(class_id, "flat")
        score = float(proba[idx]) if hasattr(proba, "__len__") else float(proba)
        return Prediction(action=action, score=score, meta=meta)
    except Exception as exc:  # pragma: no cover - network or model failure
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status != 404 and "404" not in str(exc):
            logger.error("Failed to load regime model for %s: %s", symbol, exc)
        return _baseline_action(features)


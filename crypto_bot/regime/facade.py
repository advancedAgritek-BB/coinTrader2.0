from __future__ import annotations

from typing import Any
import os

from .registry import load_latest_regime

# Cache loaded models keyed by trading symbol to avoid repeated downloads or
# deserialisation work for callers that invoke ``predict`` multiple times for
# the same asset.
_cache: dict[str, Any] = {}


def _get_model(symbol: str) -> Any:
    """Return a cached model for ``symbol`` or load it if missing."""
    if symbol not in _cache:
        _cache[symbol] = load_latest_regime(symbol)
    return _cache[symbol]


def predict(
    features_df, symbol: str = os.getenv("CT_SYMBOL", "XRPUSD")
):
    """Predict the regime action for the given ``features_df``.

    The function delegates to a LightGBM style model loaded via
    :func:`load_latest_regime`.  The predicted probabilities are converted
    into the action space ``[-1, 0, 1]`` representing short, neutral and
    long positions respectively.  A small object with ``action`` and
    ``score`` attributes is returned to remain backward compatible with
    previous implementations.
    """
    model = _get_model(symbol)
    proba = model.predict_proba(features_df)
    idx = proba.argmax(axis=1)[0]
    classes = [-1, 0, 1]
    action = classes[idx]
    score = float(proba[0, idx])
    return type("Pred", (), {"action": action, "score": score})

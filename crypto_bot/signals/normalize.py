import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Dict


def _to_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _to_str(val: Any, default: str = "none") -> str:
    try:
        return str(val)
    except Exception:
        return default


def _ensure_meta(val: Any) -> Dict[str, Any]:
    if isinstance(val, Mapping):
        return dict(val)
    return {}


def normalize_strategy_result(res: Any) -> Dict[str, Any]:
    """Canonicalise strategy outputs.

    Parameters
    ----------
    res:
        Raw result returned by a strategy which may be of various legacy shapes.

    Returns
    -------
    dict
        A dictionary with ``score`` (float), ``signal`` (str) and ``meta`` (dict)
        keys. ``meta`` may contain a ``reason`` describing any fallbacks or
        parse issues.
    """

    meta: Dict[str, Any] = {}

    if res is None:
        return {"score": 0.0, "signal": "none", "meta": meta}

    # Dataclass instances – convert to dict and recurse
    if dataclasses.is_dataclass(res):
        return normalize_strategy_result(dataclasses.asdict(res))

    # Mapping / dict-like objects
    if isinstance(res, Mapping):
        score = _to_float(res.get("score"))
        signal = _to_str(res.get("signal") or res.get("direction"))
        meta = _ensure_meta(res.get("meta"))
        reason = res.get("reason")
        if reason is not None and "reason" not in meta:
            meta["reason"] = reason
        return {"score": score, "signal": signal or "none", "meta": meta}

    # Tuple/list legacy shapes
    if isinstance(res, Sequence) and not isinstance(res, (str, bytes)):
        seq = list(res)
        score = _to_float(seq[0]) if len(seq) > 0 else 0.0
        signal = _to_str(seq[1]) if len(seq) > 1 else "none"
        if len(seq) >= 3:
            third = seq[2]
            if isinstance(third, Mapping):
                meta = dict(third)
            else:
                meta["reason"] = _to_str(third)
        if len(seq) >= 4:
            fourth = seq[3]
            if isinstance(fourth, Mapping):
                meta.update(dict(fourth))
            else:
                meta["extra"] = fourth
        return {"score": score, "signal": signal, "meta": meta}

    # Numeric score only
    if isinstance(res, (int, float)):
        return {"score": float(res), "signal": "none", "meta": meta}

    # Object with score/signal attributes
    score_attr = getattr(res, "score", None)
    signal_attr = getattr(res, "signal", None)
    meta_attr = getattr(res, "meta", None)
    if score_attr is not None or signal_attr is not None or meta_attr is not None:
        score = _to_float(score_attr)
        signal = _to_str(signal_attr)
        meta = _ensure_meta(meta_attr)
        return {"score": score, "signal": signal or "none", "meta": meta}

    # Unrecognised return type
    meta["reason"] = "unrecognized_return"
    return {"score": 0.0, "signal": "none", "meta": meta}

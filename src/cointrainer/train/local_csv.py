from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import os
import time
import pandas as pd
import numpy as np

from cointrainer.io.csv7 import read_csv7
from cointrainer.features.simple_indicators import ema, rsi, atr, roc, obv
from sklearn.preprocessing import StandardScaler

try:
    # Optional at import time; actual training imports happen inside train()
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore

@dataclass
class TrainConfig:
    symbol: str = "XRPUSD"
    horizon: int = 15                   # bars
    hold: float = 0.0015                # 0.15%
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    random_state: int = 42
    outdir: Path = Path("local_models")
    write_predictions: bool = True
    publish_to_registry: bool = False   # if True and env is present, save to registry too

FEATURE_LIST = ["ema_8","ema_21","rsi_14","atr_14","roc_5","obv"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    X = pd.DataFrame(index=df.index)
    X["ema_8"] = ema(c, 8)
    X["ema_21"] = ema(c, 21)
    X["rsi_14"] = rsi(c, 14)
    X["atr_14"] = atr(h, l, c, 14)
    X["roc_5"] = roc(c, 5)
    X["obv"] = obv(c, v)
    return X

def make_labels(close: pd.Series, horizon: int, hold: float) -> pd.Series:
    future_ret = close.pct_change(horizon).shift(-horizon)
    y = np.where(future_ret >  hold,  1, np.where(future_ret < -hold, -1, 0))
    return pd.Series(y, index=close.index)

def _fit_model(X: pd.DataFrame, y: pd.Series):
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not installed. Install with: pip install lightgbm")
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        objective="multiclass", class_weight="balanced",
        n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    return model

def _save_local(model, scaler, cfg: TrainConfig, metadata: Dict) -> Path:
    import joblib
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    path = cfg.outdir / f"{cfg.symbol.lower()}_regime_lgbm.pkl"
    joblib.dump({"model": model, "scaler": scaler}, path)
    # Save an adjacent metadata snapshot for reference
    (cfg.outdir / f"{cfg.symbol.lower()}_metadata.json").write_text(pd.Series(metadata).to_json())
    return path

def _maybe_publish_registry(model_bytes: bytes, metadata: Dict, cfg: TrainConfig) -> Optional[str]:
    if not cfg.publish_to_registry:
        return None
    try:
        # lazy import to avoid runtime deps
        from cointrainer import registry
        ts = time.strftime("%Y%m%d-%H%M%S")
        key = f"regime/{cfg.symbol}/{ts}_regime_lgbm.pkl"
        registry.save_model(key, model_bytes, metadata)
        return key
    except Exception:
        return None

def train_from_csv7(csv_path: Path | str, cfg: TrainConfig) -> Tuple[object, Dict]:
    df = read_csv7(csv_path)
    X_all = make_features(df).dropna()
    y_all = make_labels(df.loc[X_all.index, "close"], cfg.horizon, cfg.hold)
    m = y_all.notna()
    X = X_all[m]
    y = y_all[m]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=X.columns
    )
    model = _fit_model(X_scaled, y)

    metadata = {
        "schema_version": "1",
        "feature_list": FEATURE_LIST,
        "label_order": [-1, 0, 1],
        "horizon": f"{cfg.horizon}m",
        "thresholds": {"hold": cfg.hold},
        "symbol": cfg.symbol,
    }

    # Save local
    local_path = _save_local(model, scaler, cfg, metadata)

    # Optional registry publish
    try:
        import io, joblib
        buf = io.BytesIO()
        joblib.dump({"model": model, "scaler": scaler}, buf)
        _maybe_publish_registry(buf.getvalue(), metadata, cfg)
    except Exception:
        pass

    # Optional predictions CSV for inspection
    if cfg.write_predictions:
        try:
            proba = model.predict_proba(X.values)
            idx = proba.argmax(axis=1)
            index_to_class = [-1, 0, 1]
            classes = [index_to_class[i] for i in idx]
            score = proba.max(axis=1)
            out = pd.DataFrame(index=X.index)
            out["class"] = classes
            out["action"] = pd.Series(classes, index=out.index).map({-1:"short",0:"flat",1:"long"})
            out["score"] = score
            out_path = cfg.outdir / f"{cfg.symbol.lower()}_predictions.csv"
            out.to_csv(out_path, index=True)
        except Exception:
            pass

    return model, metadata

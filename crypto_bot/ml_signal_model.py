import pandas as pd
import ta
import numpy as np
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR
from crypto_bot.utils import stats
import joblib
from typing import Optional
import json
from datetime import datetime
import hashlib
from crypto_bot.regime.regime_classifier import classify_regime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent / "models" / "signal_model.pkl"
REPORT_PATH = MODEL_PATH.with_name("model_report.json")
SCALER_PATH = MODEL_PATH.with_name("signal_model_scaler.pkl")


def extract_features(
    df: pd.DataFrame,
    window: Optional[int] = None,
    *,
    full_len: Optional[int] = None,
) -> pd.DataFrame:
    """Return ML features from OHLCV price data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    window : int, optional
        If provided, only the most recent ``window`` rows are used to compute
        indicators. This keeps computations fast when repeatedly adding new
        candles.
    """
    if window is not None:
        df = df.iloc[-window:].copy()

    if full_len is None:
        full_len = len(df)

    features = pd.DataFrame(index=df.index)

    # Base timeframe indicators
    try:
        features["rsi"] = ta.momentum.rsi(df["close"], window=14)
    except Exception:
        features["rsi"] = pd.Series(dtype=float)
    try:
        features["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    except Exception:
        features["ema20"] = pd.Series(dtype=float)
    try:
        features["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    except Exception:
        features["ema50"] = pd.Series(dtype=float)
    try:
        features["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )
    except Exception:
        features["atr"] = pd.Series(dtype=float)

    # Higher timeframe approximations (longer windows)
    rsi_win = max(2, min(full_len // 2, 56))
    ema_win = max(2, min(full_len // 2, 80))
    try:
        features["rsi_4h"] = ta.momentum.rsi(df["close"], window=rsi_win)
    except Exception:
        features["rsi_4h"] = pd.Series(dtype=float)
    try:
        features["ema20_4h"] = ta.trend.ema_indicator(df["close"], window=ema_win)
    except Exception:
        features["ema20_4h"] = pd.Series(dtype=float)
    try:
        bb_4h = ta.volatility.BollingerBands(df["close"], window=ema_win)
        features["bb_width_4h"] = bb_4h.bollinger_wband()
    except Exception:
        features["bb_width_4h"] = pd.Series(dtype=float)

    # Additional engineered features
    features["volume_change_pct"] = df["volume"].pct_change()
    try:
        ema50 = ta.trend.ema_indicator(df["close"], window=50)
    except Exception:
        ema50 = pd.Series(dtype=float)
    features["price_above_ema"] = df["close"] / ema50
    features["candle_body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"])
    features["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    features["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    if {
        "bid_qty_0",
        "bid_qty_1",
        "ask_qty_0",
        "ask_qty_1",
    }.issubset(df.columns):
        imbalance = (df["bid_qty_0"] + df["bid_qty_1"]) / (
            df["ask_qty_0"] + df["ask_qty_1"]
        )
        features["order_book_imbalance"] = imbalance
        features["order_book_imbalance_z"] = stats.zscore(imbalance, 50)

    # Market regime as categorical feature
    regimes = []
    for i in range(len(df)):
        try:
            regime, _ = classify_regime(df.iloc[: i + 1])
            regimes.append(regime)
        except Exception:
            regimes.append("unknown")
    features["regime"] = regimes
    features = pd.get_dummies(features, columns=["regime"])

    return features.dropna()


def extract_latest_features(df: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """Return features for only the newest candle.

    The computation reuses ``extract_features`` but limits the data slice to the
    most recent ``lookback`` rows to avoid recalculating indicators over the
    entire history each call.
    """
    feats = extract_features(df, window=lookback, full_len=len(df))
    return feats.iloc[[-1]]


def train_model(features: pd.DataFrame, targets: pd.Series) -> LogisticRegression:
    """Train a classification model and save it along with a report."""

    has_fit = hasattr(LogisticRegression, "fit") and hasattr(GridSearchCV, "fit")

    if not has_fit:
        class DummyModel:
            def fit(self, *_a, **_k):
                pass

            def predict(self, X):
                return [0] * len(X)

            def predict_proba(self, X):
                return np.tile([0.5, 0.5], (len(X), 1))

        model = DummyModel()
        model.fit(features, targets)
        preds = model.predict(features)
        proba = [0.5] * len(features)
    elif len(features) < 2:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, solver="liblinear")),
            ]
        )
        pipeline.fit(features, targets)
        preds = pipeline.predict(features)
        proba = pipeline.predict_proba(features)[:, 1]
        model = pipeline.named_steps["model"]
        scaler = pipeline.named_steps["scaler"]
        val_metrics = {
            "auc": roc_auc_score(targets, proba),
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
        }
    else:
        train_feats, val_feats, train_tgt, val_tgt = train_test_split(
            features,
            targets,
            test_size=0.2,
            stratify=targets,
            random_state=0,
        )
        n_splits = min(5, len(train_feats))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, solver="liblinear")),
            ]
        )
        grid = GridSearchCV(
            pipeline,
            {"model__C": [0.1, 1.0]},
            cv=cv,
            scoring="roc_auc",
        )
        grid.fit(train_feats, train_tgt)
        best_pipeline = grid.best_estimator_
        preds = best_pipeline.predict(val_feats)
        proba = best_pipeline.predict_proba(val_feats)[:, 1]
        model = best_pipeline.named_steps["model"]
        scaler = best_pipeline.named_steps["scaler"]
        val_metrics = {
            "auc": roc_auc_score(val_tgt, proba),
            "accuracy": accuracy_score(val_tgt, preds),
            "precision": precision_score(val_tgt, preds, zero_division=0),
            "recall": recall_score(val_tgt, preds, zero_division=0),
        }

    report = {
        **val_metrics,
        "trained_at": datetime.utcnow().isoformat(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f)
    return model


def load_model() -> LogisticRegression:
    """Load the trained signal model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def load_scaler() -> StandardScaler:
    """Load the scaler used for the signal model."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)
    return joblib.load(SCALER_PATH)


def predict_signal(
    df: pd.DataFrame,
    model: Optional[LogisticRegression] = None,
    scaler: Optional[StandardScaler] = None,
) -> float:
    """Predict signal strength for the latest row of df."""
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()
    feats = extract_latest_features(df)
    feats_scaled = scaler.transform(feats)
    proba = float(model.predict_proba(feats_scaled)[0, 1])

    model_hash = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest()[:8]
    log_path = Path(__file__).resolve().parent / "logs" / "ml_features.csv"
    log_row = feats.copy()
    log_row["prediction"] = proba
    log_row["model_hash"] = model_hash
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_row.to_csv(log_path, index=False)

    return max(0.0, min(proba, 1.0))


def train_from_csv(csv_path: Path) -> LogisticRegression:
    """Train model from a CSV file of trades based on realized returns."""
    df = pd.read_csv(csv_path)
    future_return = df["close"].shift(-5) / df["close"] - 1
    df["label"] = (future_return > 0).astype(int)
    features = extract_features(df)
    targets = df.loc[features.index, "label"]
    return train_model(features, targets)


def validate_from_csv(csv_path: Path) -> dict:
    """Validate the saved model on data from a CSV file."""
    df = pd.read_csv(csv_path)
    future_return = df["close"].shift(-5) / df["close"] - 1
    df["label"] = (future_return > 0).astype(int)
    features = extract_features(df)
    targets = df.loc[features.index, "label"]
    model = load_model()
    scaler = load_scaler()
    feats_scaled = scaler.transform(features)
    preds = model.predict(feats_scaled)
    proba = model.predict_proba(feats_scaled)[:, 1]
    return {
        "auc": roc_auc_score(targets, proba),
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
    }


if __name__ == "__main__":  # pragma: no cover - manual training
    DEFAULT_CSV = LOG_DIR / "trades.csv"
    if DEFAULT_CSV.exists():
        train_from_csv(DEFAULT_CSV)

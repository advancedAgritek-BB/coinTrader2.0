import pandas as pd
import ta
from pathlib import Path
import joblib
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor

MODEL_PATH = Path(__file__).resolve().parent / "models" / "signal_model.pkl"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ML features from OHLCV price data."""
    features = pd.DataFrame(index=df.index)
    features["rsi"] = ta.momentum.rsi(df["close"], window=14)
    features["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    features["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    features["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    features["returns"] = df["close"].pct_change()
    features["volume"] = df["volume"]
    return features.dropna()


def train_model(features: pd.DataFrame, targets: pd.Series) -> GradientBoostingRegressor:
    """Train a gradient boosting model and save it."""
    model = GradientBoostingRegressor()
    model.fit(features, targets)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> GradientBoostingRegressor:
    """Load the trained signal model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def predict_signal(df: pd.DataFrame, model: Optional[GradientBoostingRegressor] = None) -> float:
    """Predict signal strength for the latest row of df."""
    if model is None:
        model = load_model()
    feats = extract_features(df).iloc[[-1]]
    score = float(model.predict(feats)[0])
    return max(0.0, min(score, 1.0))


def train_from_csv(csv_path: Path) -> GradientBoostingRegressor:
    """Train model from a CSV file of trades with ``signal_strength`` column."""
    df = pd.read_csv(csv_path)
    if "signal_strength" not in df.columns:
        raise ValueError("CSV must contain signal_strength column")
    features = extract_features(df)
    targets = df.loc[features.index, "signal_strength"]
    return train_model(features, targets)


if __name__ == "__main__":  # pragma: no cover - manual training
    DEFAULT_CSV = Path("crypto_bot/logs/trades.csv")
    if DEFAULT_CSV.exists():
        train_from_csv(DEFAULT_CSV)

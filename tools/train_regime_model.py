import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from supabase import create_client


LOG = logging.getLogger("train_regime_model")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

TRADES_PATH = Path("crypto_bot/logs/trades.csv")
MODEL_PATH = Path("xrpusd_regime_lgbm.pkl")


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOG.error("Trade log not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "symbol" not in df.columns or "price" not in df.columns:
        LOG.error("Unexpected trade log format")
        return pd.DataFrame()
    return df


def select_high_vol_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["symbol"].str.contains("SOL", na=False)].copy()
    symbols = df["symbol"].unique()
    high_vol = []
    for sym in symbols:
        sub = df[df["symbol"] == sym]
        pct = sub["price"].pct_change().abs()
        if pct.std() > 0.05 or sym == "FARTCOIN/SOL":
            high_vol.append(sym)
    LOG.info("Selected symbols: %s", high_vol)
    return df[df["symbol"].isin(high_vol)]


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values("timestamp")
    df["return"] = df.groupby("symbol")["price"].pct_change()
    df["target"] = (df["return"].rolling(5).mean().shift(-1) > 0).astype(int)
    feats = df[["return"]].fillna(0)
    targ = df["target"].fillna(0)
    return feats, targ


def train_model(X: pd.DataFrame, y: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)
    LOG.info("Saved model to %s", path)


def upload_to_supabase(path: Path) -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        LOG.error("Missing Supabase credentials")
        return
    client = create_client(url, key)
    with open(path, "rb") as f:
        try:
            client.storage.from_("models").upload(path.name, f, {
                "content-type": "application/octet-stream",
                "upsert": True,
            })
            LOG.info("Uploaded %s to Supabase", path)
        except Exception:
            LOG.exception("Failed to upload %s", path)


def main() -> None:
    trades = load_trades(TRADES_PATH)
    if trades.empty:
        LOG.error("No data for training")
        return
    trades = select_high_vol_pairs(trades)
    X, y = build_dataset(trades)
    model = train_model(X, y)
    save_model(model, MODEL_PATH)
    upload_to_supabase(MODEL_PATH)


if __name__ == "__main__":
    main()

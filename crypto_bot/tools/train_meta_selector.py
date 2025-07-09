import json
from pathlib import Path
from typing import Dict, List

from crypto_bot.utils.logger import LOG_DIR


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
import pickle

LOG_FILE = LOG_DIR / "strategy_performance.json"
MODEL_FILE = Path("crypto_bot/models/meta_selector_lgbm.pkl")
K = 20  # number of trades for label window


def _compute_features(trades: List[Dict[str, float]]) -> Dict[str, float] | None:
    if not trades:
        return None
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    wins = sum(p > 0 for p in pnls)
    total = len(pnls)
    win_rate = wins / total if total else 0.0
    series = pd.Series(pnls)
    neg = series[series < 0]
    downside_std = neg.std(ddof=0) if not neg.empty else 0.0
    max_dd = (series.cummax() - series).max()
    raw_sharpe = 0.0
    std = series.std()
    if std:
        raw_sharpe = series.mean() / std * np.sqrt(total)
    return {
        "win_rate": win_rate,
        "raw_sharpe": float(raw_sharpe),
        "downside_std": float(downside_std),
        "max_dd": float(max_dd),
        "trade_count": total,
    }


def build_dataset(path: Path = LOG_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    rows = []
    for regime, strat_map in data.items():
        if not isinstance(strat_map, dict):
            continue
        for strategy, trades in strat_map.items():
            if not isinstance(trades, list) or len(trades) < 2 * K:
                continue
            for i in range(len(trades) - 2 * K + 1):
                feat_slice = trades[i : i + K]
                label_slice = trades[i + K : i + 2 * K]
                features = _compute_features(feat_slice)
                if features is None:
                    continue
                label = sum(float(t.get("pnl", 0.0)) for t in label_slice)
                rows.append({
                    "regime": regime,
                    "strategy": strategy,
                    **features,
                    "label": label,
                })
    return pd.DataFrame(rows)


def train(df: pd.DataFrame) -> lgb.LGBMRegressor:
    X = df[["win_rate", "raw_sharpe", "downside_std", "max_dd", "trade_count"]]
    y = df["label"]
    model = lgb.LGBMRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Perform cross-validation to evaluate
    cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    model.fit(X, y)
    return model


def main() -> None:
    df = build_dataset()
    if df.empty:
        raise ValueError("No training data found")
    model = train(df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()

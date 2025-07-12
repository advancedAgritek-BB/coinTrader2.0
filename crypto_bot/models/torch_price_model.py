from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import ta

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "price_model.pt"


class PriceNet(nn.Module):  # pragma: no cover - tiny network
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


_model: Optional[nn.Module] = None


def load_model() -> nn.Module:
    """Load model weights from :data:`MODEL_PATH`."""
    global _model
    if _model is None:
        if torch is None:
            raise ImportError("torch not available")
        model = PriceNet()
        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH)
                model.load_state_dict(state)
            except Exception:  # pragma: no cover - corrupted state
                pass
        model.eval()
        _model = model
    return _model


def train_model(df_cache: Dict[str, Dict[str, pd.DataFrame]]) -> Optional[nn.Module]:
    """Train a simple price prediction model using ``df_cache['1h']``."""
    if torch is None or not hasattr(nn.Module, "parameters"):
        # Torch missing or stubbed; create empty model file for consistency
        if torch is not None and hasattr(torch, "save"):
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save({}, MODEL_PATH)
            except Exception:  # pragma: no cover - stub save may fail
                pass
        return None
    tf_map = df_cache.get("1h")
    if not isinstance(tf_map, dict):
        return None
    dfs = [df for df in tf_map.values() if isinstance(df, pd.DataFrame) and not df.empty]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).copy()
    df["momentum_rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["trend_macd"] = ta.trend.macd(df["close"])
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    features = df[[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "momentum_rsi",
        "trend_macd",
    ]].values
    targets = df["target"].values
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = PriceNet()
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(10):  # pragma: no cover - simple training loop
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return model


def predict_price(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    """Predict the next close price for ``df``."""
    if torch is None:
        return float("nan")
    if model is None:
        model = load_model()
    if df.empty:
        return float("nan")
    df = df.copy()
    df["momentum_rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["trend_macd"] = ta.trend.macd(df["close"])
    row = df[[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "momentum_rsi",
        "trend_macd",
    ]].iloc[[-1]].values
    x = torch.tensor(row, dtype=torch.float32)
    with torch.no_grad():
        val = model(x).squeeze().item()
    return float(val)

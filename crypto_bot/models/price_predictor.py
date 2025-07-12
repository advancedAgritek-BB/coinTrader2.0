from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "price_predictor.pt"


class PriceNet(nn.Module):  # pragma: no cover - tiny network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


_model: Optional[nn.Module] = None


def load_model() -> nn.Module:
    """Load the price prediction model."""
    global _model
    if _model is None:
        if torch is None:
            raise ImportError("torch not available")
        model = PriceNet()
        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH)
                if hasattr(model, "load_state_dict"):
                    model.load_state_dict(state)
            except Exception:  # pragma: no cover - corrupted state
                pass
        if hasattr(model, "eval"):
            model.eval()
        _model = model
    return _model


def predict_price(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    """Predict next closing price for ``df``."""
    if torch is None:
        return float("nan")
    if model is None:
        model = load_model()
    if df.empty:
        return float("nan")
    row = df[["open", "high", "low", "close"]].iloc[[-1]].values
    x = torch.tensor(row, dtype=torch.float32)
    if hasattr(torch, "no_grad"):
        with torch.no_grad():
            val = model(x).squeeze().item()
    else:
        val = model(x).squeeze().item()
    return float(val)


def predict_score(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    """Return scaled score based on predicted price change."""
    if df.empty or "close" not in df.columns:
        return 0.0
    pred = predict_price(df, model=model)
    last = float(df["close"].iloc[-1])
    if last == 0 or not isinstance(pred, float):
        return 0.0
    score = (pred - last) / last
    return max(0.0, min(score, 1.0))

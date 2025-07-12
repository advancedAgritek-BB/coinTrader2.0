from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "models" / "torch_price_model.pt"


class PriceNet(nn.Module):  # pragma: no cover - small network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(df: pd.DataFrame, epochs: int = 20) -> Optional[nn.Module]:
    """Train ``PriceNet`` on OHLCV data."""
    if torch is None:
        return None
    if df.empty:
        return None
    feats = df[["open", "high", "low", "close", "volume"]].iloc[:-1]
    targets = df["close"].shift(-1).dropna()
    X = torch.tensor(feats.values, dtype=torch.float32)
    y = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = PriceNet()
    if not hasattr(model, "parameters"):
        return None
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):  # pragma: no cover - simple loop
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return model


def load_model() -> nn.Module:
    if torch is None:
        raise ImportError("torch not available")
    model = PriceNet()
    state = torch.load(MODEL_PATH)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_price(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    if torch is None:
        return float("nan")
    if model is None:
        model = load_model()
    if df.empty:
        return float("nan")
    row = df[["open", "high", "low", "close", "volume"]].iloc[[-1]].values
    x = torch.tensor(row, dtype=torch.float32)
    with torch.no_grad():
        val = float(model(x).squeeze().item())
    return val

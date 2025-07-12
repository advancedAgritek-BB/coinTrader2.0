import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "models" / "torch_price_model.pt"
REPORT_PATH = MODEL_PATH.with_name("torch_price_model_report.json")


class PriceNet(nn.Module):  # pragma: no cover - simple network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def _build_dataset(df_cache: Dict[str, Dict[str, pd.DataFrame]]):
    """Return tensors built from OHLCV cache."""
    X_parts = []
    y_parts = []
    for tf_cache in df_cache.values():
        for df in tf_cache.values():
            if df is None or df.empty:
                continue
            if len(df) < 2:
                continue
            arr = df[["open", "high", "low", "close"]].values
            X_parts.append(arr[:-1])
            y_parts.append(df["close"].values[1:])
    if not X_parts:
        return None, None
    import numpy as np

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


def train_model(df_cache: Dict[str, Dict[str, pd.DataFrame]], epochs: int = 5) -> Optional[nn.Module]:
    """Train a simple network to predict next closing price."""
    if torch is None:
        return None
    X, y = _build_dataset(df_cache)
    if X is None:
        return None
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = PriceNet()
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):  # pragma: no cover - training loop
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    report = {"trained_at": datetime.utcnow().isoformat()}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f)
    return model

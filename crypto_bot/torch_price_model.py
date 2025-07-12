import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
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
REPORT_PATH = MODEL_PATH.with_name("torch_price_model_report.json")


    from torch import nn
except Exception:  # pragma: no cover - allow running without torch
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "models" / "torch_price_model.pt"


class PriceNet(nn.Module):  # pragma: no cover - small network
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
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
    report = {"trained_at": datetime.utcnow().isoformat()}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f)
    return model
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
_model: Optional[nn.Module] = None


def load_model() -> nn.Module:
    """Load the torch price prediction model."""
    global _model
    if _model is None:
        if torch is None:
            raise ImportError("torch not available")
        model = PriceNet()
        if MODEL_PATH.exists():
            try:  # pragma: no cover - best effort loading
                state = torch.load(MODEL_PATH)
                if hasattr(model, "load_state_dict"):
                    model.load_state_dict(state)
            except Exception:
                pass
        if hasattr(model, "eval"):
            model.eval()
        _model = model
    return _model


def predict_price(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    """Predict next closing price for ``df`` using the torch model."""
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
    row = df[["open", "high", "low", "close"]].iloc[[-1]].values
    x = torch.tensor(row, dtype=torch.float32)
    if hasattr(torch, "no_grad"):
        with torch.no_grad():
            val = model(x).squeeze().item()
    else:
        val = model(x).squeeze().item()
    return float(val)

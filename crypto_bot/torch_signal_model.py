import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from crypto_bot.ml_signal_model import extract_features, extract_latest_features

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None

MODEL_PATH = Path(__file__).resolve().parent / "models" / "torch_model.pt"
REPORT_PATH = MODEL_PATH.with_name("torch_model_report.json")


class SimpleNet(nn.Module):  # pragma: no cover - small network
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(features: pd.DataFrame, targets: pd.Series, epochs: int = 20) -> Optional[nn.Module]:
    """Train a tiny neural network on the given features."""
    if torch is None:
        return None
    X = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = SimpleNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):  # pragma: no cover - simple training loop
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


def load_model() -> nn.Module:
    if torch is None:
        raise ImportError("torch not available")
    model_state = torch.load(MODEL_PATH)
    dummy = SimpleNet(len(model_state["net.0.weight"][0]))
    dummy.load_state_dict(model_state)
    dummy.eval()
    return dummy


def predict_signal(df: pd.DataFrame, model: Optional[nn.Module] = None) -> float:
    """Predict signal strength for the latest row of df."""
    if torch is None:
        return 0.0
    if model is None:
        model = load_model()
    feats = extract_latest_features(df)
    x = torch.tensor(feats.values, dtype=torch.float32)
    with torch.no_grad():
        pred = float(model(x).squeeze().clamp(0, 1).item())
    return pred


def train_from_csv(csv_path: Path) -> Optional[nn.Module]:
    df = pd.read_csv(csv_path)
    future_return = df["close"].shift(-5) / df["close"] - 1
    df["label"] = (future_return > 0).astype(int)
    features = extract_features(df)
    targets = df.loc[features.index, "label"]
    return train_model(features, targets)


if __name__ == "__main__":  # pragma: no cover - manual training helper
    DEFAULT_CSV = MODEL_PATH.parent.parent / "logs" / "trades.csv"
    if DEFAULT_CSV.exists():
        train_from_csv(DEFAULT_CSV)

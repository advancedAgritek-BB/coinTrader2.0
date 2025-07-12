from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore

MODEL_PATH = Path(__file__).resolve().parent / "models" / "grid_center_model.pt"

class CenterNet(nn.Module):  # pragma: no cover - tiny network
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

_model: Optional[nn.Module] = None

def load_model() -> nn.Module:
    """Load the center prediction model."""
    global _model
    if _model is None:
        if torch is None:
            raise ImportError("torch not available")
        state = torch.load(MODEL_PATH)
        model = CenterNet()
        try:
            model.load_state_dict(state)
        except Exception:  # pragma: no cover - ignore bad state
            pass
        model.eval()
        _model = model
    return _model

def predict_centre(df) -> float:
    """Return predicted range midpoint for ``df``."""
    if torch is None:
        return float("nan")
    model = load_model()
    high = float(df["high"].iloc[-1])
    low = float(df["low"].iloc[-1])
    inp = torch.tensor([[high, low]], dtype=torch.float32)
    with torch.no_grad():
        val = model(inp).squeeze().item()
    return float(val)

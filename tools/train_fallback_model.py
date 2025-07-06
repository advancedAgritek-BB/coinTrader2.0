"""Train and export the gradient boosting fallback model."""

import base64
import io

import joblib
import numpy as np
from lightgbm import LGBMClassifier


def main(out_path: str = "model_base64.txt") -> None:
    """Train a simple model and export it as base64."""
    np.random.seed(0)
    X = np.random.randn(2000, 1)
    y = (X[:, 0] > 0).astype(int)
    model = LGBMClassifier(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    with open(out_path, "w") as f:
        f.write(encoded)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

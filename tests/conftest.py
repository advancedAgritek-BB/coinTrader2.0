import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import importlib.util
try:  # optional numpy dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy not installed
    import numpy_stub as np
    sys.modules.setdefault("numpy", np)

try:  # optional pandas dependency
    import pandas as pd
    from pandas import Series
except Exception:  # pragma: no cover - pandas not installed
    import pandas_stub as pd
    from pandas_stub import Series
    sys.modules.setdefault("pandas", pd)
    sys.modules.setdefault("pandas.core", pd)
import pytest

if importlib.util.find_spec("pytest_asyncio") is not None:
    pytest_plugins = ("pytest_asyncio",)
else:  # pragma: no cover - plugin optional
    pytest_plugins = ()

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Skip network tests when HTTP libraries are unavailable."""
    def _has_module(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    if not _has_module("requests") and not _has_module("aiohttp"):
        skip_net = pytest.mark.skip(reason="network dependencies not installed")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_net)


# Provide a minimal stub for the optional ``telegram`` package so modules that
# depend on it can be imported without the real dependency. If the real package
# is installed we prefer to use it so related tests can run.
try:  # pragma: no cover - optional dependency
    import telegram  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - telegram not installed
    class _FakeTelegram:
        class Bot:
            def __init__(self, *a, **k):
                pass

    sys.modules.setdefault("telegram", _FakeTelegram())


# Lightweight stub for ``scipy`` used in some utility modules.
class _FakeStats:
    @staticmethod
    def pearsonr(a, b):
        return 0.0, 0.0


class _FakeScipy:
    stats = _FakeStats()


sys.modules.setdefault("scipy", _FakeScipy())
sys.modules.setdefault("scipy.stats", _FakeStats())


# ``joblib`` is imported by the ML model but isn't required for these tests.
class _FakeJoblib:
    def __init__(self):
        self.storage = {}

    def load(self, path):
        return self.storage.get(str(path))

    def dump(self, obj, path):
        self.storage[str(path)] = obj
        Path(path).touch()


sys.modules.setdefault("joblib", _FakeJoblib())


# Minimal stub for the ``sklearn`` package required by ml_signal_model.
class _FakeSklearn:
    class linear_model:
        class LogisticRegression:
            def __init__(self, *a, **k):
                pass

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.zeros((len(X), 2))

    class preprocessing:
        class StandardScaler:
            def fit(self, *a, **k):
                return self

            def fit_transform(self, X, *a, **k):
                return X

            def transform(self, X, *a, **k):
                return X

    class pipeline:
        class Pipeline:
            def __init__(self, steps):
                self.steps = steps

            @property
            def named_steps(self):
                return {name: obj for name, obj in self.steps}

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.zeros((len(X), 2))

    class model_selection:
        class StratifiedKFold:
            def __init__(self, *a, **k):
                pass

        class GridSearchCV:
            def __init__(self, *a, **k):
                self.best_estimator_ = _FakeSklearn.pipeline.Pipeline([
                    ("scaler", _FakeSklearn.preprocessing.StandardScaler()),
                    ("model", _FakeSklearn.linear_model.LogisticRegression()),
                ])

            def fit(self, X, y):
                return self

        def train_test_split(X, y, *a, **k):
            return X, X, y, y

    class metrics:
        @staticmethod
        def accuracy_score(*args, **kwargs):
            return 0.0

        @staticmethod
        def precision_score(*args, **kwargs):
            return 0.0

        @staticmethod
        def recall_score(*args, **kwargs):
            return 0.0

        @staticmethod
        def roc_auc_score(*args, **kwargs):
            return 0.0


sys.modules.setdefault("sklearn", _FakeSklearn())
sys.modules.setdefault("sklearn.linear_model", _FakeSklearn.linear_model)
sys.modules.setdefault("sklearn.model_selection", _FakeSklearn.model_selection)
sys.modules.setdefault("sklearn.metrics", _FakeSklearn.metrics)
sys.modules.setdefault("sklearn.preprocessing", _FakeSklearn.preprocessing)
sys.modules.setdefault("sklearn.pipeline", _FakeSklearn.pipeline)

# Minimal stub for ``torch`` used by the PyTorch signal model.
class _FakeTorch:
    float32 = None

    @staticmethod
    def tensor(data, dtype=None):
        class _T(float):
            def unsqueeze(self, _):
                return self

            def squeeze(self):
                return self

            def clamp(self, _a, _b):
                return self

            def item(self):
                return 0.0

        return _T(0.0)

    class nn:
        class Module:
            pass

        class Linear(Module):
            def __init__(self, *a, **k):
                pass

            def __call__(self, x):
                return _FakeTorch.tensor(0.0)

        class Sequential(Module):
            def __init__(self, *a, **k):
                pass

            def __call__(self, x):
                return _FakeTorch.tensor(0.0)

        class ReLU(Module):
            pass

        class Sigmoid(Module):
            pass

        class BCELoss(Module):
            def __call__(self, *a, **k):
                return _FakeTorch.tensor(0.0)

    class optim:
        class Adam:
            def __init__(self, *a, **k):
                pass

            def zero_grad(self):
                pass

            def step(self):
                pass

    class utils:
        class data:
            class TensorDataset:
                def __init__(self, *a, **k):
                    pass

            class DataLoader:
                def __init__(self, *a, **k):
                    pass

                def __iter__(self):
                    yield _FakeTorch.tensor(0.0), _FakeTorch.tensor(0.0)

    @staticmethod
    def save(_obj, path):
        Path(path).touch()

    @staticmethod
    def load(_path):
        return {}


sys.modules.setdefault("torch", _FakeTorch())
sys.modules.setdefault("torch.nn", _FakeTorch.nn)
sys.modules.setdefault("torch.optim", _FakeTorch.optim)
sys.modules.setdefault("torch.utils.data", _FakeTorch.utils.data)

# Minimal stub for the optional ``ta`` package used in ML signal models.
try:  # pragma: no cover - optional dependency
    import ta  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - ta not installed
    class _FakeTa:
        class trend:
            @staticmethod
            def macd(*args, **kwargs):
                return 0.0

            @staticmethod
            def ema_indicator(series, window=14):
                return Series([80.0] * len(series))

        class momentum:
            @staticmethod
            def rsi(series, window=14):
                values = list(series)
                rsis = [50.0]
                for i in range(1, len(values)):
                    start = max(0, i - window + 1)
                    window_vals = values[start : i + 1]
                    gains = [max(window_vals[j] - window_vals[j - 1], 0) for j in range(1, len(window_vals))]
                    losses = [max(window_vals[j - 1] - window_vals[j], 0) for j in range(1, len(window_vals))]
                    avg_gain = sum(gains) / max(len(gains), 1)
                    avg_loss = sum(losses) / max(len(losses), 1)
                    rs = avg_gain / avg_loss if avg_loss else 100
                    rsi_val = 100 - 100 / (1 + rs)
                    rsis.append(rsi_val)
                return Series(rsis)

        class volatility:
            @staticmethod
            def average_true_range(high, low, close, window=14):
                return Series([1.0] * len(close))

    sys.modules.setdefault("ta", _FakeTa())

# Lightweight stub for PyYAML
class _FakeYaml:
    @staticmethod
    def safe_load(*args, **kwargs):
        return {}

sys.modules.setdefault("yaml", _FakeYaml())

# Basic stub for aiohttp
class _FakeAioHttp:
    class ClientSession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def get(self, *a, **k):
            class R:
                status = 200

                async def json(self):
                    return {}

                async def text(self):
                    return ""

            return R()

sys.modules.setdefault("aiohttp", _FakeAioHttp())

# Basic ``requests`` stub
try:  # pragma: no cover - optional dependency
    import requests  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - requests not installed
    class _FakeRequests:
        class Response:
            status_code = 200

            def json(self):
                return {}

            @property
            def text(self):
                return ""

        def get(self, *a, **k):
            return self.Response()

        class Session:
            def get(self, *a, **k):
                return _FakeRequests.Response()

            def close(self):
                pass

sys.modules.setdefault("requests", _FakeRequests())

# Lightweight stubs for additional optional dependencies used in various modules.
class _FakeLimiter:
    class AsyncLimiter:
        def __init__(self, *a, **k):
            pass

sys.modules.setdefault("aiolimiter", _FakeLimiter())

class _FakeCcxt:
    class ExchangeError(Exception):
        pass

    class RequestTimeout(Exception):
        pass

    class BadSymbol(Exception):
        pass

sys.modules.setdefault("ccxt", _FakeCcxt())
sys.modules.setdefault("ccxt.async_support", _FakeCcxt())

class _FakePydantic:
    class ValidationError(Exception):
        pass

sys.modules.setdefault("pydantic", _FakePydantic())


@pytest.fixture(autouse=True)
def _clear_strategy_router_cache():
    """Ensure strategy router caches are cleared between tests."""
    try:
        import crypto_bot.strategy_router as sr
    except Exception:
        # Module may not be importable in minimal test environments
        yield
        return

    sr._build_mappings_cached.cache_clear()
    sr._CONFIG_REGISTRY.clear()
    yield
    sr._build_mappings_cached.cache_clear()
    sr._CONFIG_REGISTRY.clear()

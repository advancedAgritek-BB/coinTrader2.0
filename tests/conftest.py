import sys
import importlib.util
import numpy as np
from pathlib import Path
from pandas import Series
import pytest

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

            @staticmethod
            def adx(high, low, close, window=14):
                return Series([50.0] * len(close))

            class ADXIndicator:
                def __init__(self, *a, **k):
                    pass

                def adx(self):
                    return Series([20.0])

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

            class BollingerBands:
                def __init__(self, series, window=14, window_dev=2):
                    self.series = series

                def bollinger_wband(self):
                    return Series([0.05] * len(self.series))

            class KeltnerChannel:
                def __init__(self, high, low, close, window=14):
                    self.series = close

                def keltner_channel_hband(self):
                    return Series([1.0] * len(self.series))

                def keltner_channel_lband(self):
                    return Series([0.0] * len(self.series))

    sys.modules.setdefault("ta", _FakeTa())
    sys.modules.setdefault("ta.trend", _FakeTa.trend)
    sys.modules.setdefault("ta.momentum", _FakeTa.momentum)
    sys.modules.setdefault("ta.volatility", _FakeTa.volatility)

# Lightweight stub for PyYAML
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - PyYAML not installed
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

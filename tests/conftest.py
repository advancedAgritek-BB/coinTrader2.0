import importlib.util
import sys
import types
from pathlib import Path
import pytest

try:  # Ensure real redis module is loaded for tests using fakeredis
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy not installed
    class _FakeRandomState:
        def __init__(self, seed=None):
            self.seed = seed

        def rand(self, *shape):
            if not shape:
                return 0.0
            if len(shape) == 1:
                return [0.0] * shape[0]
            return [[0.0] * shape[1] for _ in range(shape[0])]

    def _zeros(shape, dtype=None):
        if isinstance(shape, int):
            return [0.0] * shape
        if isinstance(shape, tuple) and len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0]

    def _isnan(x):
        return x != x

    class ndarray(list):
        pass

    import math

    def _sqrt(x):
        return math.sqrt(x)

    def _std(_x):
        return 0.0

    def _corrcoef(a, b):
        return [[1.0, 1.0], [1.0, 1.0]]

    def _percentile(_a, _p):
        return 0.0

    def _isscalar(_x):
        return not isinstance(_x, (list, tuple, dict))

    np = types.SimpleNamespace(
        random=types.SimpleNamespace(RandomState=_FakeRandomState),
        zeros=_zeros,
        isnan=_isnan,
        ndarray=ndarray,
        sqrt=_sqrt,
        std=_std,
        corrcoef=_corrcoef,
        percentile=_percentile,
        isscalar=_isscalar,
        nan=float("nan"),
    )
    sys.modules.setdefault("numpy", np)

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
    from pandas import Series  # type: ignore
except Exception:  # pragma: no cover - pandas not installed
    class DataFrame(dict):
        def __init__(self, data=None):
            data = data or {}
            super().__init__(data)
            self.data = {k: list(v) for k, v in data.items()}
            self.columns = list(self.data.keys())
            self.__dict__.update(self.data)
            self.index = list(range(len(next(iter(self.data.values()), []))))

        def __getitem__(self, key):
            return self.data[key]

        def __len__(self):
            return len(next(iter(self.data.values()), []))

        @property
        def empty(self):
            return len(self) == 0

        class _ILoc:
            def __init__(self, outer):
                self.outer = outer

            def __getitem__(self, idx):
                if isinstance(idx, int) and idx < 0:
                    idx = len(self.outer) + idx
                return {k: v[idx] for k, v in self.outer.data.items()}

            def __setitem__(self, idx, value_map):
                if isinstance(idx, int) and idx < 0:
                    idx = len(self.outer) + idx
                for k, v in value_map.items():
                    self.outer.data.setdefault(k, [None] * len(self.outer))
                    self.outer.data[k][idx] = v

        @property
        def iloc(self):
            return self._ILoc(self)

        class _Loc(_ILoc):
            def __getitem__(self, key):
                idx, col = key
                if isinstance(idx, int) and idx < 0:
                    idx = len(self.outer) + idx
                return self.outer.data[col][idx]

            def __setitem__(self, key, value):
                idx, col = key
                if isinstance(idx, int) and idx < 0:
                    idx = len(self.outer) + idx
                self.outer.data[col][idx] = value

        @property
        def loc(self):
            return self._Loc(self)

        def pct_change(self):
            return self

        def to_numpy(self):
            return [list(v) for v in zip(*self.data.values())] if self.data else []

        def dropna(self):
            return self

        def std(self):
            return 0.0

        def mean(self):
            return 0.0

        def cumsum(self):
            return self

    class Series(list):
        def tail(self, n):
            return Series(self[-n:])

        def to_numpy(self):
            return list(self)

        def std(self):
            return 0.0

        def mean(self):
            return 0.0

        def cummax(self):
            return self

    def _isnan_pd(x):
        return x != x

    class _Timedelta:
        def __init__(self, value=None, **kwargs):
            seconds = 0.0
            if isinstance(value, str):
                import re
                m = re.match(r"(\d+)([smhd])", value.strip().lower())
                if m:
                    n = int(m.group(1))
                    unit = m.group(2)
                    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
                    seconds = n * scale
                else:
                    try:
                        seconds = float(value)
                    except Exception:
                        seconds = 0.0
            elif isinstance(value, (int, float)):
                seconds = float(value)
            seconds += kwargs.get("seconds", 0) * 1
            seconds += kwargs.get("minutes", 0) * 60
            seconds += kwargs.get("hours", 0) * 3600
            seconds += kwargs.get("days", 0) * 86400
            self._seconds = seconds

        def total_seconds(self):
            return float(self._seconds)

    pd = types.SimpleNamespace(
        DataFrame=DataFrame,
        Series=Series,
        Timedelta=_Timedelta,
        isna=_isnan_pd,
        isnan=_isnan_pd,
    )
    sys.modules.setdefault("pandas", pd)

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide minimal stubs for optional dependencies
import types
sys.modules.setdefault("ccxt.pro", types.ModuleType("ccxt.pro"))
# Basic ccxt stub so utils can import without the real dependency
_ccxt_mod = types.ModuleType("ccxt")
sys.modules.setdefault("ccxt", _ccxt_mod)
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))
sys.modules.setdefault("base58", types.ModuleType("base58"))
class _FakeProm:
    class Counter:
        def __init__(self, *a, **k):
            pass

        def inc(self, *_a, **_k):
            pass

sys.modules.setdefault("prometheus_client", _FakeProm())


@pytest.fixture
def progress_path(tmp_path):
    path = tmp_path / "ohlcv_progress.log"
    yield path
    if path.exists():
        path.unlink()


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


# Stubs to allow importing strategy_router without optional dependencies.
class _FakeTelegram:
    class TelegramNotifier:
        def __init__(self, *a, **k):
            pass

        async def notify_async(self, text):
            pass

        def notify(self, text):
            pass

    @staticmethod
    def send_test_message(*_a, **_k):
        pass

    @staticmethod
    def send_message(*_a, **_k):
        pass

    @staticmethod
    def send_message_sync(*_a, **_k):
        pass


sys.modules.setdefault("crypto_bot.utils.telegram", _FakeTelegram())

# Prepopulate common token decimals to avoid network lookups in tests
try:
    from crypto_bot.utils.token_registry import TOKEN_DECIMALS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TOKEN_DECIMALS = {}
TOKEN_DECIMALS.setdefault("SOL", 9)
TOKEN_DECIMALS.setdefault("USDC", 6)


class _FakeSolanaMempool:
    class SolanaMempoolMonitor:
        pass


sys.modules.setdefault(
    "crypto_bot.execution.solana_mempool", _FakeSolanaMempool()
)


sniper_solana_mod = types.ModuleType("crypto_bot.strategies.sniper_solana")


def _fake_sniper_signal(df, config=None):  # pragma: no cover - simple stub
    return 0.0, "none"


sniper_solana_mod.generate_signal = _fake_sniper_signal
sys.modules.setdefault("crypto_bot.strategies.sniper_solana", sniper_solana_mod)


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
    class WSMsgType:
        TEXT = "text"
        CLOSED = "closed"
        ERROR = "error"

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


@pytest.fixture(autouse=True)
def _mock_listing_date(monkeypatch):
    """Avoid network calls for Kraken listing timestamps."""
    try:
        from crypto_bot.utils import market_loader
    except Exception:  # pragma: no cover - optional dependency
        yield
        return

    async def _no_listing(*_a, **_k):
        return None

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", _no_listing)
    yield


@pytest.fixture()
def regime_pnl_file(tmp_path, monkeypatch):
    """Provide a temp regime PnL log without seeded trades."""
    from crypto_bot.utils import regime_pnl_tracker as rpt

    log = tmp_path / "regime_pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)
    monkeypatch.setattr(rpt, "_seed_fake_trades", lambda *a, **k: None)
    return log

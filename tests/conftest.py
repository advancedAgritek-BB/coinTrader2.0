import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# Provide a minimal stub for the optional ``telegram`` package so modules that
# depend on it can be imported without the real dependency. If the real package
# is installed we prefer to use it so related tests can run.
try:  # pragma: no cover - optional dependency
    import telegram  # type: ignore
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

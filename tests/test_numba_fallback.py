import numpy as np
from crypto_bot.utils import symbol_scoring as sc


def test_score_vectorised_fallback(monkeypatch):
    import pandas as pd
    vol = np.array([1000.0, 500.0])
    chg = np.array([1.0, -2.0])
    spr = np.array([0.1, 0.2])
    liq = np.array([0.5, 0.2])
    df = pd.DataFrame({"vol": vol, "chg": chg, "spr": spr, "liq": liq})
    cfg = {"use_numba_scoring": True}

    monkeypatch.setattr(sc, "HAS_NUMBA", False, raising=False)
    monkeypatch.setattr(sc, "_score_vectorised_numba", lambda *a, **k: (_ for _ in ()).throw(AssertionError()), raising=False)

    res = sc.score_vectorised(df, cfg)
    expected = sc.score_vectorised(df, {"use_numba_scoring": False})
    assert np.allclose(res, expected)

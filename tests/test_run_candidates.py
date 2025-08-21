import asyncio
import asyncio
import pandas as pd
import crypto_bot.utils.market_analyzer as ma


def strat_a(df, cfg=None):
    return 0.4, "long"


def strat_b(df, cfg=None):
    return 0.5, "short"


def strat_c(df, cfg=None):
    return 0.2, "long"


def test_run_candidates_ranking(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    scores = {strat_a: (0.4, "long"), strat_b: (0.5, "short"), strat_c: (0.2, "long")}

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        return [(*scores[s], None) for s in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    edges = {"strat_a": 0.5, "strat_b": 1.0, "strat_c": 2.0}
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: edges[name])

    res = asyncio.run(ma.run_candidates(df, [strat_a, strat_b, strat_c], "AAA", {}, "trending"))
    names = [fn.__name__ for fn, _s, _d in res]
    assert names == ["strat_b", "strat_c", "strat_a"]


def test_regime_filter_zero_scores(monkeypatch):
    df = pd.DataFrame({"close": [1, 1]})

    def a(df, cfg=None):
        return 0.0, "none"

    def b(df, cfg=None):
        return 0.0, "none"

    class AFilt:
        @staticmethod
        def matches(r):
            return r == "trending"

    class BFilt:
        @staticmethod
        def matches(r):
            return False

    a.regime_filter = AFilt()
    b.regime_filter = BFilt()

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        return [(0.0, "none", None) for _ in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: 1.0)

    res = asyncio.run(ma.run_candidates(df, [b, a], "AAA", {}, "trending"))
    assert res[0][0] is a


def test_regime_filter_equal_scores(monkeypatch):
    df = pd.DataFrame({"close": [1, 1]})

    def a(df, cfg=None):
        return 0.5, "long"

    def b(df, cfg=None):
        return 0.5, "long"

    class AFilt:
        @staticmethod
        def matches(r):
            return r == "trending"

    class BFilt:
        @staticmethod
        def matches(r):
            return False

    a.regime_filter = AFilt()
    b.regime_filter = BFilt()

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        return [(0.5, "long", None) for _ in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: 1.0)

    res = asyncio.run(ma.run_candidates(df, [b, a], "AAA", {}, "trending"))
    assert res[0][0] is a


def test_min_accept_score(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})

    def low(df, cfg=None):
        return 0.4, "long"

    def high(df, cfg=None):
        return 0.7, "long"

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        scores = {low: (0.4, "long"), high: (0.7, "long")}
        return [(*scores[s], None) for s in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: 1.0)

    cfg = {"router": {"min_accept_score": 0.5}}
    res = asyncio.run(ma.run_candidates(df, [low, high], "AAA", cfg, "trending"))
    names = [fn.__name__ for fn, _s, _d in res]
    assert names == ["high"]


def test_fast_track(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})

    def fast(df, cfg=None):
        return 0.96, "long"

    def normal(df, cfg=None):
        return 0.8, "long"

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        scores = {fast: (0.96, "long"), normal: (0.8, "long")}
        return [(*scores[s], None) for s in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: 1.0)

    res = asyncio.run(ma.run_candidates(df, [normal, fast], "AAA", {}, "trending"))
    assert len(res) == 1 and res[0][0] is fast


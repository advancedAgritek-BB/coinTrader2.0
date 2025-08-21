from dataclasses import dataclass, field
from typing import Any, Dict

from crypto_bot.signals.normalize import normalize_strategy_result


@dataclass
class Dummy:
    score: float
    signal: str
    meta: Dict[str, Any] = field(default_factory=dict)


def test_dict_result():
    res = normalize_strategy_result({"score": 1, "signal": "buy", "meta": {"foo": "bar"}})
    assert res == {"score": 1.0, "signal": "buy", "meta": {"foo": "bar"}}


def test_dataclass_result():
    d = Dummy(2.0, "sell", {"why": "test"})
    res = normalize_strategy_result(d)
    assert res == {"score": 2.0, "signal": "sell", "meta": {"why": "test"}}


def test_tuple_legacy():
    res2 = normalize_strategy_result((3, "hold"))
    res3 = normalize_strategy_result((4, "buy", "because"))
    res4 = normalize_strategy_result((5, "sell", "because", {"foo": "bar"}))
    assert res2 == {"score": 3.0, "signal": "hold", "meta": {}}
    assert res3 == {"score": 4.0, "signal": "buy", "meta": {"reason": "because"}}
    assert res4 == {"score": 5.0, "signal": "sell", "meta": {"reason": "because", "foo": "bar"}}


def test_unrecognized_type():
    res = normalize_strategy_result(object())
    assert res["score"] == 0.0
    assert res["signal"] == "none"
    assert res["meta"].get("reason") == "unrecognized_return"

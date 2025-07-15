import sys
import types
import importlib.util
from pathlib import Path

sys.modules.setdefault("telegram", types.SimpleNamespace(Bot=None))
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: ""))
sys.modules.setdefault("ccxt", types.SimpleNamespace())
sys.modules.setdefault("aiohttp", types.SimpleNamespace(ClientSession=None))
sys.modules.setdefault("cachetools", types.SimpleNamespace(TTLCache=dict))
sys.modules.setdefault("prometheus_client", types.SimpleNamespace(Counter=lambda *a, **k: None))

spec = importlib.util.spec_from_file_location(
    "dex_scalper", Path(__file__).resolve().parents[1] / "crypto_bot/strategy/dex_scalper.py"
)
dex_scalper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dex_scalper)


def test_priority_fee_from_env(monkeypatch):
    monkeypatch.setenv("MOCK_ETH_PRIORITY_FEE_GWEI", "12.5")
    assert dex_scalper.fetch_priority_fee_gwei() == 12.5


def test_priority_fee_rpc(monkeypatch):
    monkeypatch.delenv("MOCK_ETH_PRIORITY_FEE_GWEI", raising=False)
    called = {}

    def fake_post(url, json=None, timeout=5):
        called["url"] = url
        called["payload"] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "result": {
                        "reward": [
                            ["0x3b9aca00"],
                            ["0x77359400"],
                            ["0xb2d05e00"],
                            ["0xee6b2800"],
                            ["0x12a05f200"],
                        ]
                    }
                }

        return Resp()

    monkeypatch.setattr(dex_scalper.requests, "post", fake_post)
    fee = dex_scalper.fetch_priority_fee_gwei("http://rpc")
    assert fee == 3.0
    assert called["url"] == "http://rpc"
    assert called["payload"]["method"] == "eth_feeHistory"


def test_priority_fee_error(monkeypatch):
    monkeypatch.delenv("MOCK_ETH_PRIORITY_FEE_GWEI", raising=False)

    def fake_post(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(dex_scalper.requests, "post", fake_post)
    assert dex_scalper.fetch_priority_fee_gwei("http://rpc") == 0.0


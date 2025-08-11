import sys
import types

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))

from frontend import api
from fastapi.testclient import TestClient
import json
import pytest

def test_live_signals_endpoint(tmp_path, monkeypatch):
    scores = {"BTC": 0.5, "ETH": 0.1}
    f = tmp_path / "scores.json"
    f.write_text(json.dumps(scores))
    monkeypatch.setattr(api, "SIGNALS_FILE", f)
    client = TestClient(api.app)
    resp = client.get("/live-signals")
    assert resp.status_code == 200
    assert resp.json() == scores


def test_positions_endpoint(tmp_path, monkeypatch):
    log = tmp_path / "pos.log"
    log.write_text(
        "2023-01-01 00:00:00 - INFO - Active XBT/USDT buy 1 entry 100 current 110 pnl $10 (positive) balance $1100\n"
    )
    monkeypatch.setattr(api, "POSITIONS_FILE", log)
    client = TestClient(api.app)
    resp = client.get("/positions")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["symbol"] == "XBT/USDT"
    assert data[0]["pnl"] == 10


def test_strategy_performance_endpoint(tmp_path, monkeypatch):
    data = {"trend": {"bot": [{"symbol": "BTC", "pnl": 1}]}}
    f = tmp_path / "perf.json"
    f.write_text(json.dumps(data))
    monkeypatch.setattr(api, "PERFORMANCE_FILE", f)
    client = TestClient(api.app)
    resp = client.get("/strategy-performance")
    assert resp.status_code == 200
    assert resp.json() == data


def test_strategy_scores_endpoint(tmp_path, monkeypatch):
    scores = {"trend_bot": {"sharpe": 1.0}}
    f = tmp_path / "scores.json"
    f.write_text(json.dumps(scores))
    monkeypatch.setattr(api, "SCORES_FILE", f)
    client = TestClient(api.app)
    resp = client.get("/strategy-scores")
    assert resp.status_code == 200
    assert resp.json() == scores


def test_reload_config_endpoint(monkeypatch):
    class Dummy:
        def __init__(self):
            self.called = False

        async def reload_config(self):
            self.called = True
            return {"status": "reloaded"}

    dummy = Dummy()
    monkeypatch.setattr(api, "CONTROLLER", dummy)
    client = TestClient(api.app)
    resp = client.post("/reload-config")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"
    assert dummy.called is True


def test_close_all_endpoint(monkeypatch):
    class Dummy:
        def __init__(self):
            self.called = False
            self.state = {}

        async def close_all_positions(self):
            self.called = True
            self.state["liquidate_all"] = True
            return {"status": "liquidation_scheduled"}

    dummy = Dummy()
    monkeypatch.setattr(api, "CONTROLLER", dummy)
    client = TestClient(api.app)
    resp = client.post("/close-all")
    assert resp.status_code == 200
    assert resp.json()["status"] == "liquidation_scheduled"
    assert dummy.called is True
    assert dummy.state.get("liquidate_all") is True


@pytest.mark.asyncio
async def test_close_all_positions_writes_to_proc(monkeypatch):
    class DummyStdin:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

        async def drain(self):
            pass

    class DummyProc:
        def __init__(self, stdin):
            self.stdin = stdin
            self.returncode = None

    stdin = DummyStdin()
    proc = DummyProc(stdin)
    from crypto_bot.bot_controller import TradingBotController

    controller = TradingBotController.__new__(TradingBotController)
    controller.proc = proc
    controller.state = {}

    result = await controller.close_all_positions()
    assert stdin.writes == [b"panic sell\n"]
    assert result["status"] == "command_sent"

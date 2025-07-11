import json
from crypto_bot.utils import performance_logger as pl
from crypto_bot.selector.bandit import Bandit
import types


def test_log_performance_grouping(tmp_path, monkeypatch):
    log = tmp_path / "perf.json"
    monkeypatch.setattr(pl, "LOG_FILE", log)

    calls = []

    def update(symbol, strategy, win):
        calls.append((symbol, strategy, win))

    dummy_bandit = types.SimpleNamespace(update=update)
    monkeypatch.setattr(pl, "bandit", dummy_bandit)

    rec1 = {
        "symbol": "SOL/USDC",
        "regime": "trending",
        "strategy": "trend_bot",
        "pnl": 1.0,
        "entry_time": "e1",
        "exit_time": "x1",
    }
    rec2 = {
        "symbol": "SOL/USDC",
        "regime": "trending",
        "strategy": "trend_bot",
        "pnl": 2.0,
        "entry_time": "e2",
        "exit_time": "x2",
    }
    pl.log_performance(rec1)
    pl.log_performance(rec2)

    assert calls == [
        ("SOL/USDC", "trend_bot", True),
        ("SOL/USDC", "trend_bot", True),
    ]

    data = json.loads(log.read_text())
    assert "trending" in data
    assert "trend_bot" in data["trending"]
    assert len(data["trending"]["trend_bot"]) == 2
    assert data["trending"]["trend_bot"][0]["pnl"] == 1.0
    assert data["trending"]["trend_bot"][1]["pnl"] == 2.0


def test_log_performance_bandit_counter(tmp_path, monkeypatch):
    log = tmp_path / "perf.json"
    monkeypatch.setattr(pl, "LOG_FILE", log)

    b = Bandit(state_file=tmp_path / "bandit.json")
    monkeypatch.setattr(pl, "bandit", b)

    rec = {
        "symbol": "XBT/USDT",
        "regime": "trending",
        "strategy": "trend_bot",
        "pnl": 1.0,
        "entry_time": "e",
        "exit_time": "x",
    }

    for _ in range(50):
        pl.log_performance(rec)

    assert b.update_count == 50
    assert b.state_file.exists()


def test_log_performance_sniper_solana(tmp_path, monkeypatch):
    log = tmp_path / "perf.json"
    monkeypatch.setattr(pl, "LOG_FILE", log)

    dummy_bandit = types.SimpleNamespace(update=lambda *a, **k: None)
    monkeypatch.setattr(pl, "bandit", dummy_bandit)

    rec = {
        "symbol": "SOL/USDC",
        "regime": "volatile",
        "strategy": "sniper_solana",
        "pnl": 0.5,
        "entry_time": "e",
        "exit_time": "x",
    }

    pl.log_performance(rec)

    data = json.loads(log.read_text())
    assert data["volatile"]["sniper_solana"][0]["pnl"] == 0.5

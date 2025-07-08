import types
from crypto_bot import main
from crypto_bot.utils import regime_pnl_tracker as rpt

class DummyNotifier:
    def __init__(self):
        self.msg = None
    def notify(self, text):
        self.msg = text


def test_maybe_update_mode(monkeypatch):
    state = {"mode": "onchain"}
    cfg = {"mode_threshold": 0.5, "mode_degrade_window": 3}
    notifier = DummyNotifier()

    monkeypatch.setattr(main, "get_recent_win_rate", lambda w: 0.4)
    main.maybe_update_mode(state, "onchain", cfg, notifier)
    assert state["mode"] == "cex"
    assert "cex" in notifier.msg

    monkeypatch.setattr(main, "get_recent_win_rate", lambda w: 0.6)
    main.maybe_update_mode(state, "onchain", cfg, notifier)
    assert state["mode"] == "onchain"


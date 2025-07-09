from crypto_bot.solana.risk import RiskTracker


def test_concurrent_limit(tmp_path):
    risk_file = tmp_path / "risk.json"
    tracker = RiskTracker(risk_file)
    tracker.add_snipe("M1", 1)
    cfg = {"max_concurrent": 1}
    assert tracker.allow_snipe("M2", cfg) is False


def test_daily_loss_cap_persistence(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    path = state_dir / "risk.json"
    tracker = RiskTracker(path)
    tracker.add_snipe("M1", 1)
    tracker.record_loss("M1", 5.0)

    new_tracker = RiskTracker(path)
    assert new_tracker.state.daily_loss == 5.0
    cfg = {"daily_loss_cap": 4.0}
    assert new_tracker.allow_snipe("M2", cfg) is False

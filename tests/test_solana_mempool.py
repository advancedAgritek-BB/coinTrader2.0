from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


def test_is_suspicious_from_env(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "150")
    monitor = SolanaMempoolMonitor()
    assert monitor.is_suspicious(100) is True


def test_is_not_suspicious(monkeypatch):
    monkeypatch.setenv("MOCK_PRIORITY_FEE", "10")
    monitor = SolanaMempoolMonitor()
    assert monitor.is_suspicious(100) is False

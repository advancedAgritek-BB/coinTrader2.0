import json
import fakeredis

from crypto_bot.utils import commit_lock


def test_check_and_update_no_previous_regime(monkeypatch):
    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(commit_lock, "REDIS_CLIENT", fake_r)
    result = commit_lock.check_and_update("sideways", 60, 3)
    assert result == "sideways"
    stored = json.loads(fake_r.get(commit_lock.REDIS_KEY))
    assert stored["regime"] == "sideways"


def test_check_and_update_enforces_duration(monkeypatch):
    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(commit_lock, "REDIS_CLIENT", fake_r)
    assert commit_lock.check_and_update("trending", 60, 3) == "trending"
    first = json.loads(fake_r.get(commit_lock.REDIS_KEY))
    result = commit_lock.check_and_update("sideways", 60, 3)
    assert result == "trending"
    assert json.loads(fake_r.get(commit_lock.REDIS_KEY)) == first

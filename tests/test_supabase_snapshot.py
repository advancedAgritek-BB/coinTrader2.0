import sys
import types

import crypto_bot.utils.supabase_snapshot as ss


class FakeBucket:
    def __init__(self, data):
        self.data = data
        self.path = None

    def download(self, path):
        self.path = path
        return self.data


class FakeStorage:
    def __init__(self, bucket):
        self.bucket = bucket
        self.name = None

    def from_(self, name):
        self.name = name
        return self.bucket


class FakeClient:
    def __init__(self, bucket):
        self.storage = FakeStorage(bucket)


def test_fetch_snapshot_success(monkeypatch):
    bucket = FakeBucket(b'{"a": 1}')
    fake_client = FakeClient(bucket)
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setitem(
        sys.modules,
        "supabase",
        types.SimpleNamespace(create_client=lambda u, k: fake_client),
    )

    snap = ss.fetch_snapshot("mint")
    assert snap == {"a": 1}
    assert bucket.path == "mint.json"
    assert fake_client.storage.name == "snapshots"


def test_fetch_snapshot_failure(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    class BadClient:
        def __init__(self):
            class S:
                def from_(self, name):
                    class B:
                        def download(self, path):
                            raise Exception("fail")
                    return B()
            self.storage = S()

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        types.SimpleNamespace(create_client=lambda u, k: BadClient()),
    )

    snap = ss.fetch_snapshot("mint")
    assert snap is None


"""Minimal stub of the ``fakeredis`` package used in tests."""

from __future__ import annotations

from typing import Any, Dict
import types


class FakeRedis:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def flushall(self) -> None:  # pragma: no cover - trivial
        self._data.clear()

    def lock(self, *args: Any, **kwargs: Any):  # pragma: no cover - trivial
        class _Lock:
            def acquire(self, *a, **k):
                return True

            def release(self, *a, **k):
                pass

        return _Lock()


FakeStrictRedis = FakeRedis
aioredis = types.SimpleNamespace(FakeRedis=FakeRedis)

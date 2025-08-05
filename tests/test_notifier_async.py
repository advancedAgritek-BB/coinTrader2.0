import asyncio
import time

import pytest

from crypto_bot.utils.notifier import Notifier
import crypto_bot.utils.notifier as notifier_mod


@pytest.mark.asyncio
async def test_notify_async_success(monkeypatch):
    calls = []

    def fake_send(token, chat, text):
        calls.append(text)
        return None

    monkeypatch.setattr(notifier_mod, "send_message", fake_send)
    n = Notifier("t", "c")
    res = await n.notify_async("hello")
    assert res is None
    assert calls == ["hello"]


def test_notify_sync_wrapper(monkeypatch):
    calls = []

    def fake_send(token, chat, text):
        calls.append(text)
        return None

    monkeypatch.setattr(notifier_mod, "send_message", fake_send)
    n = Notifier("t", "c")
    res = n.notify("hi")
    assert res is None
    assert calls == ["hi"]


@pytest.mark.asyncio
async def test_notify_inside_event_loop(monkeypatch):
    calls = []

    def fake_send(token, chat, text):
        calls.append(text)
        return None

    monkeypatch.setattr(notifier_mod, "send_message", fake_send)
    n = Notifier("t", "c")
    res = n.notify("loop")
    assert res is None
    assert calls == ["loop"]


@pytest.mark.asyncio
async def test_value_error_reporting(monkeypatch):
    def fake_send(token, chat, text):
        raise ValueError("bad token")

    monkeypatch.setattr(notifier_mod, "send_message", fake_send)
    n = Notifier("t", "c")
    res = await n.notify_async("msg")
    assert "bad token" in res
    res2 = n.notify("msg")
    assert "bad token" in res2


@pytest.mark.asyncio
async def test_timeout_retries(monkeypatch):
    calls = 0

    def slow_send(token, chat, text):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return None

    monkeypatch.setattr(notifier_mod, "send_message", slow_send)
    n = Notifier("t", "c")
    res = await n.notify_async("slow", retries=2, timeout=0.01)
    assert res == "Timed out after 2 attempts"
    assert calls == 2


def test_sync_timeout_retries(monkeypatch):
    calls = 0

    def fail_send(token, chat, text):
        nonlocal calls
        calls += 1
        raise TimeoutError("network")

    monkeypatch.setattr(notifier_mod, "send_message", fail_send)
    n = Notifier("t", "c")
    res = n.notify("slow", retries=2)
    assert res == "Timed out after 2 attempts"
    assert calls == 2


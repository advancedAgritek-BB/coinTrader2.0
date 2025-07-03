import asyncio
import warnings

from crypto_bot.utils.telegram import send_message


def test_send_message_async_running(monkeypatch):
    calls = {}

    class DummyBot:
        def __init__(self, token):
            calls['token'] = token

        async def send_message(self, chat_id, text):
            calls['chat_id'] = chat_id
            calls['text'] = text

    monkeypatch.setattr('crypto_bot.utils.telegram.Bot', DummyBot)

    async def runner():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('error')
            err = send_message('t', 'c', 'msg')
        return err, w

    err, w = asyncio.run(runner())

    assert err is None
    assert calls['chat_id'] == 'c'
    assert calls['text'] == 'msg'
    assert w == []

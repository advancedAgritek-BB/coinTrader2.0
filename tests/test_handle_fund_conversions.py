import asyncio

import crypto_bot.main as main


class DummyNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, text: str):
        self.messages.append(text)


def test_handle_fund_conversions_to_btc(monkeypatch):
    called = {}

    async def fake_convert(wallet, from_t, to_t, amt, **kwargs):
        called['convert'] = (from_t, to_t, amt)
        return {}

    monkeypatch.setattr(main, 'auto_convert_funds', fake_convert)
    monkeypatch.setattr(main, 'check_wallet_balances', lambda w: {'ETH': 1.0})
    monkeypatch.setattr(main, 'detect_non_trade_tokens', lambda b: ['ETH'])

    async def fake_balance(ex, cfg):
        return 1.2345

    monkeypatch.setattr(main, 'get_btc_balance', fake_balance)

    def fake_check(new_balance, reason, *, currency='USDT'):
        called['balance'] = (new_balance, reason, currency)

    notifier = DummyNotifier()
    asyncio.run(
        main.handle_fund_conversions(
            object(),
            {'execution_mode': 'dry_run'},
            notifier,
            'wallet',
            fake_check,
        )
    )

    assert called['convert'] == ('ETH', 'BTC', 1.0)
    assert called['balance'] == (1.2345, 'funds converted', 'BTC')
    assert notifier.messages == ['Converted to 1.234500 BTC']

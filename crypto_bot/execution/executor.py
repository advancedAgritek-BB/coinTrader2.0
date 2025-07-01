import ccxt
from typing import Dict

from crypto_bot.utils.telegram import send_message
from crypto_bot.utils.trade_logger import log_trade


def load_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    return exchange


def execute_trade(exchange: ccxt.Exchange, symbol: str, side: str, amount: float, config: Dict, token: str, chat_id: str, dry_run: bool = True) -> None:
    pre_msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, pre_msg)

    if not dry_run:
        try:
            order = exchange.create_market_order(symbol, side, amount)
        except Exception as e:
            send_message(token, chat_id, f"Order failed: {e}")
            return
    else:
        order = {'symbol': symbol, 'side': side, 'amount': amount, 'dry_run': True}

    send_message(token, chat_id, f"Order executed: {order}")
    log_trade(order)

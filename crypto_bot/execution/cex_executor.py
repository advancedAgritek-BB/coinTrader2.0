import os
import ccxt
from typing import Dict, Optional, Tuple

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.trade_logger import log_trade



def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client."""
    exchange_name = config.get("exchange", "coinbase")
    use_ws = config.get("use_websocket", False)

    ws_client: Optional[KrakenWSClient] = None

    if exchange_name == "coinbase":
        exchange = ccxt.coinbase({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "password": os.getenv("API_PASSPHRASE"),
            "enableRateLimit": True,
        })
    elif exchange_name == "kraken":
        exchange = ccxt.kraken({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "enableRateLimit": True,
        })
        if use_ws:
            ws_client = KrakenWSClient(os.getenv("API_KEY"), os.getenv("API_SECRET"))
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    return exchange, ws_client


def execute_trade(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str,
    chat_id: str,
    dry_run: bool = True,
) -> Dict:
    msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
    else:
        try:
            if ws_client is not None:
                order = ws_client.add_order(symbol, side, amount)
            else:
                order = exchange.create_market_order(symbol, side, amount)
        except Exception as e:
            send_message(token, chat_id, f"Order failed: {e}")
            return {}
    send_message(token, chat_id, f"Order executed: {order}")
    log_trade(order)
    return order

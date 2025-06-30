import os
import json
import asyncio
import ccxt
from typing import Dict, Optional
import pandas as pd
from dotenv import dotenv_values
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from crypto_bot.utils.telegram import send_message
from crypto_bot.data.kraken_ws import KrakenWebsocketClient



def get_exchange(config):
    """Instantiate and return a ccxt exchange based on config."""
    exchange_name = config.get("exchange", "coinbase")

    if exchange_name == "coinbase":
        return ccxt.coinbase({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "password": os.getenv("API_PASSPHRASE"),
            "enableRateLimit": True,
        })
    elif exchange_name == "kraken":
        return ccxt.kraken({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "enableRateLimit": True,
        })
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")


def execute_trade(exchange: ccxt.Exchange, symbol: str, side: str, amount: float, token: str, chat_id: str, dry_run: bool = True) -> Dict:
    msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {'symbol': symbol, 'side': side, 'amount': amount, 'dry_run': True}
    else:
        try:
            order = exchange.create_market_order(symbol, side, amount)
        except Exception as e:
            send_message(token, chat_id, f"Order failed: {e}")
            return {}
    send_message(token, chat_id, f"Order executed: {order}")
    log_trade(order)
    return order


def log_trade(order: Dict) -> None:
    df = pd.DataFrame([order])
    df.to_csv('crypto_bot/logs/trades.csv', mode='a', header=False, index=False)
    try:
        creds_path = dotenv_values('crypto_bot/.env').get('GOOGLE_CRED_JSON')
        if creds_path:
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open('trade_logs').sheet1
            sheet.append_row(list(order.values()))
    except Exception:
        pass


def get_ws_token(exchange: ccxt.Exchange) -> Optional[str]:
    """Return a WebSocket authentication token if supported."""
    try:
        method = getattr(exchange, "private_post_getwebsocketstoken")
    except AttributeError:
        return None
    try:
        result = method()
        return result.get("result", {}).get("token")
    except Exception:
        return None


def place_trailing_stop_ws(
    client: KrakenWebsocketClient,
    symbol: str,
    side: str,
    qty: float,
    price_pct: float,
) -> None:
    """Send trailing-stop order via Kraken WebSocket."""
    msg = {
        "method": "add_order",
        "params": {
            "order_type": "trailing-stop",
            "side": side,
            "symbol": symbol,
            "order_qty": qty,
            "triggers": {"reference": "last", "price": price_pct, "price_type": "pct"},
            "token": client.token,
        },
    }
    if client.private_ws:
        asyncio.create_task(client.private_ws.send(json.dumps(msg)))

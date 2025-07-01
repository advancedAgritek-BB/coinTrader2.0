import os
import ccxt
from typing import Dict, Optional, Tuple
import pandas as pd
from dotenv import dotenv_values
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.kraken_ws import KrakenWSClient



def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client."""
    exchange_name = config.get("exchange", "coinbase")
    use_ws = config.get("use_websocket", False)

    ws_client: Optional[KrakenWSClient] = None
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    ws_token = os.getenv("KRAKEN_WS_TOKEN")
    api_token = os.getenv("KRAKEN_API_TOKEN")

    if exchange_name == "coinbase":
        exchange = ccxt.coinbase({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "password": os.getenv("API_PASSPHRASE"),
            "enableRateLimit": True,
        })
    elif exchange_name == "kraken":
        exchange = ccxt.kraken({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        if use_ws and ((api_key and api_secret) or ws_token):
            ws_client = KrakenWSClient(api_key, api_secret, ws_token, api_token)
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
    use_websocket: bool = False,
) -> Dict:
    if use_websocket and ws_client is None and not dry_run:
        raise ValueError("WebSocket trading enabled but ws_client is missing")
    msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
    else:
        try:
            if use_websocket:
                order = ws_client.add_order(symbol, side, amount)
            else:
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

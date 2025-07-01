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
    use_ws = config.get("use_websockets", False)

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

import ccxt
from typing import Dict
import pandas as pd
from dotenv import dotenv_values
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from crypto_bot.utils.telegram import send_message


def load_exchange(api_key: str, api_secret: str) -> ccxt.Exchange:
    return ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })


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

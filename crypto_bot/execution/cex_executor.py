import os
import time
import ccxt
from typing import Dict, Optional, Tuple, List
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
    config: Optional[Dict] = None,
) -> Dict:
    """Execute a trade with optional liquidity checks and TWAP execution."""
    config = config or {}

    def has_liquidity(order_size: float) -> bool:
        try:
            depth = config.get("liquidity_depth", 10)
            ob = exchange.fetch_order_book(symbol, limit=depth)
            book = ob["asks" if side == "buy" else "bids"]
            vol = 0.0
            for _, qty in book:
                vol += qty
                if vol >= order_size:
                    return True
            return False
        except Exception as err:
            send_message(token, chat_id, f"Order book error: {err}")
            return False

    def place(size: float) -> Dict:
        if dry_run:
            return {"symbol": symbol, "side": side, "amount": size, "dry_run": True}
        try:
            if ws_client is not None:
                return ws_client.add_order(symbol, side, size)
            return exchange.create_market_order(symbol, side, size)
        except Exception as exc:
            send_message(token, chat_id, f"Order failed: {exc}")
            return {}

    send_message(token, chat_id, f"Placing {side} order for {amount} {symbol}")

    if config.get("liquidity_check", True) and not has_liquidity(amount):
        send_message(token, chat_id, "Insufficient liquidity for order size")
        return {}

    orders: List[Dict] = []
    if config.get("twap_enabled", False) and config.get("twap_slices", 1) > 1:
        slices = config.get("twap_slices", 1)
        delay = config.get("twap_interval_seconds", 1)
        slice_amount = amount / slices
        for i in range(slices):
            if config.get("liquidity_check", True) and not has_liquidity(slice_amount):
                send_message(token, chat_id, "Insufficient liquidity during TWAP execution")
                break
            order = place(slice_amount)
            if order:
                log_trade(order)
                orders.append(order)
                send_message(token, chat_id, f"TWAP slice {i+1}/{slices} executed: {order}")
            if i < slices - 1:
                time.sleep(delay)
    else:
        order = place(amount)
        if order:
            log_trade(order)
            orders.append(order)
            send_message(token, chat_id, f"Order executed: {order}")

    return {"orders": orders}


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

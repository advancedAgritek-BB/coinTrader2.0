import os
import ccxt
try:
    import ccxt.pro as ccxtpro  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxtpro = None
from typing import Dict, Optional, Tuple

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.kraken_ws import KrakenWSClient
import asyncio
from crypto_bot.utils.trade_logger import log_trade



def get_exchange(config) -> Tuple[ccxt.Exchange, Optional[KrakenWSClient]]:
    """Instantiate and return a ccxt exchange and optional websocket client.

    When ``use_websocket`` is enabled and ``ccxtpro`` is available, an
    asynchronous ``ccxt.pro`` instance is returned. Otherwise the standard
    ``ccxt`` exchange is used. ``KrakenWSClient`` is retained for backward
    compatibility when WebSocket trading is desired without ccxt.pro.
    """

    exchange_name = config.get("exchange", "coinbase")
    use_ws = config.get("use_websocket", False)

    ws_client: Optional[KrakenWSClient] = None

    if use_ws and ccxtpro:
        ccxt_mod = ccxtpro
    else:
        ccxt_mod = ccxt

    if exchange_name == "coinbase":
        exchange = ccxt_mod.coinbase({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "password": os.getenv("API_PASSPHRASE"),
            "enableRateLimit": True,
        })
    elif exchange_name == "kraken":
        exchange = ccxt_mod.kraken({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "enableRateLimit": True,
        })
        if use_ws and not ccxtpro:
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


async def execute_trade_async(
    exchange: ccxt.Exchange,
    ws_client: Optional[KrakenWSClient],
    symbol: str,
    side: str,
    amount: float,
    token: str,
    chat_id: str,
    dry_run: bool = True,
) -> Dict:
    """Asynchronous version of :func:`execute_trade`. It supports both
    ``ccxt.pro`` exchanges and the threaded ``KrakenWSClient`` fallback."""

    msg = f"Placing {side} order for {amount} {symbol}"
    send_message(token, chat_id, msg)
    if dry_run:
        order = {"symbol": symbol, "side": side, "amount": amount, "dry_run": True}
    else:
        try:
            if ws_client is not None and not ccxtpro:
                order = ws_client.add_order(symbol, side, amount)
            elif asyncio.iscoroutinefunction(getattr(exchange, "create_market_order", None)):
                order = await exchange.create_market_order(symbol, side, amount)
            else:
                order = await asyncio.to_thread(exchange.create_market_order, symbol, side, amount)
        except Exception as e:  # pragma: no cover - network
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

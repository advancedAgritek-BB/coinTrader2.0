import pandas as pd
from typing import Dict
from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def log_trade(order: Dict, is_stop: bool = False) -> None:
    """Append executed order details to a CSV and optionally Google Sheets.

    If ``is_stop`` is ``True`` the order is recorded as a stop placement rather
    than an executed trade.
    """
    order = dict(order)
    ts = order.get("timestamp") or datetime.utcnow().isoformat()
    record = {
        "symbol": order.get("symbol", ""),
        "side": order.get("side", ""),
        "amount": order.get("amount", 0.0),
        "price": order.get("price") or order.get("average") or 0.0,
        "timestamp": ts,
        "is_stop": is_stop,
    }
    if is_stop:
        record["stop_price"] = order.get("stop") or order.get("stop_price") or 0.0

    df = pd.DataFrame([record])
    log_file = LOG_DIR / "trades.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Append rows without a header so repeated logs don't duplicate columns
    df.to_csv(log_file, mode="a", header=False, index=False)
    msg = "Stop order placed: %s" if is_stop else "Logged trade: %s"
    logger.info(msg, record)
    try:
        root_env = Path(__file__).resolve().parents[2] / '.env'
        creds_path = dotenv_values(root_env).get('GOOGLE_CRED_JSON')
        if creds_path:
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open('trade_logs').sheet1
            sheet.append_row([record[k] for k in ["symbol", "side", "amount", "price", "timestamp"]])
    except Exception:
        pass

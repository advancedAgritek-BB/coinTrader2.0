import pandas as pd
from typing import Dict
from datetime import datetime
import os
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
        creds_path = dotenv_values('crypto_bot/.env').get('GOOGLE_CRED_JSON')
        if creds_path:
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open('trade_logs').sheet1
            sheet.append_row([record[k] for k in ["symbol", "side", "amount", "price", "timestamp"]])
    except Exception:
        pass


def upload_trade_record(
    symbol: str,
    entry_price: float,
    exit_price: float,
    side: str,
    timestamp: str | None = None,
) -> None:
    """Upload a trade record to Supabase if credentials are available."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return

    try:  # pragma: no cover - optional dependency
        from supabase import create_client
    except Exception as exc:  # pragma: no cover - log import failure
        logger.error("Supabase client unavailable: %s", exc)
        return

    record = {
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "side": side,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }
    try:
        table = os.getenv("SUPABASE_TRADE_TABLE", "trade_logs")
        client = create_client(url, key)
        client.table(table).insert(record).execute()
        logger.info("Uploaded trade to Supabase: %s", record)
    except Exception as exc:
        logger.error("Failed to upload trade record: %s", exc)

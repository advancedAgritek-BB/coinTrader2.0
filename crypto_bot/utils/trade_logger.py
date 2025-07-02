import pandas as pd
from typing import Dict
from pathlib import Path
from datetime import datetime
from dotenv import dotenv_values
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def log_trade(order: Dict) -> None:
    """Append executed order details to a CSV and optionally Google Sheets."""
    order = dict(order)
    order.setdefault("timestamp", datetime.utcnow().isoformat())
    df = pd.DataFrame([order])
    log_file = Path("crypto_bot/logs/trades.csv")
    df.to_csv(log_file, mode="a", header=False, index=False)
    try:
        creds_path = dotenv_values('crypto_bot/.env').get('GOOGLE_CRED_JSON')
        if creds_path:
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
            client = gspread.authorize(creds)
            sheet = client.open('trade_logs').sheet1
            sheet.append_row(list(order.values()))
    except Exception:
        pass

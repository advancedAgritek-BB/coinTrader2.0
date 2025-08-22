from __future__ import annotations

from fastapi import FastAPI
import json
import re
from typing import TYPE_CHECKING

from crypto_bot.utils.logger import LOG_DIR
import yaml
from pathlib import Path
from wallet import Wallet
from crypto_bot.config import short_selling_enabled

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from crypto_bot.bot_controller import TradingBotController

app = FastAPI()
CONTROLLER: "TradingBotController | None" = None


def get_controller() -> "TradingBotController":
    global CONTROLLER
    if CONTROLLER is None:
        from crypto_bot.bot_controller import TradingBotController
        cfg_path = Path("crypto_bot/config.yaml")
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        wallet = None
        if cfg.get("execution_mode") == "dry_run":
            exec_cfg = cfg.get("execution", {}) or {}
            wallet = Wallet(
                cfg.get("start_balance", 1000.0),
                exec_cfg.get("max_positions", cfg.get("max_open_trades", 1)),
                short_selling_enabled(cfg),
                stake_usd=exec_cfg.get("stake_usd"),
                min_price=exec_cfg.get("min_price", 0.0),
                min_notional=exec_cfg.get("min_notional", 0.0),
            )
        CONTROLLER = TradingBotController(wallet=wallet)
    return CONTROLLER
SIGNALS_FILE = LOG_DIR / "asset_scores.json"
POSITIONS_FILE = LOG_DIR / "positions.log"
PERFORMANCE_FILE = LOG_DIR / "strategy_performance.json"
SCORES_FILE = LOG_DIR / "strategy_scores.json"


@app.get("/live-signals")
def live_signals() -> dict:
    """Return latest signal scores as a mapping of symbol to score."""
    if SIGNALS_FILE.exists():
        try:
            return json.loads(SIGNALS_FILE.read_text())
        except Exception:
            return {}
    return {}


POS_PATTERN = re.compile(
    r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
    r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
    r"pnl \$(?P<pnl>[0-9.+-]+).*balance \$(?P<balance>[0-9.]+)"
)


@app.get("/positions")
def positions() -> list[dict]:
    """Return parsed position log entries."""
    if not POSITIONS_FILE.exists():
        return []
    entries: list[dict] = []
    for line in POSITIONS_FILE.read_text().splitlines():
        match = POS_PATTERN.search(line)
        if not match:
            continue
        entries.append(
            {
                "symbol": match.group("symbol"),
                "side": match.group("side"),
                "amount": float(match.group("amount")),
                "entry_price": float(match.group("entry")),
                "current_price": float(match.group("current")),
                "pnl": float(match.group("pnl")),
                "balance": float(match.group("balance")),
            }
        )
    return entries


@app.get("/strategy-performance")
def strategy_performance() -> dict:
    """Return raw strategy performance data grouped by regime and strategy."""
    if PERFORMANCE_FILE.exists():
        try:
            return json.loads(PERFORMANCE_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.get("/strategy-scores")
def strategy_scores() -> dict:
    """Return computed strategy metrics."""
    if SCORES_FILE.exists():
        try:
            return json.loads(SCORES_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.post("/reload-config")
async def reload_config() -> dict:
    """Reload ``crypto_bot`` configuration and return status."""
    return await get_controller().reload_config()


@app.post("/close-all")
async def close_all() -> dict:
    """Request liquidation of all open positions."""
    return await get_controller().close_all_positions()

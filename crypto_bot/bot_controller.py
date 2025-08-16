from __future__ import annotations

"""Async wrapper controlling the trading bot."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

from .paper_wallet import PaperWallet
from wallet import Wallet

from crypto_bot.utils.logger import LOG_DIR
from crypto_bot import main


import yaml
from crypto_bot.utils.symbol_utils import fix_symbol

from .portfolio_rotator import PortfolioRotator
from .console_monitor import get_open_trades
from .execution.cex_executor import get_exchange, execute_trade_async


class TradingBotController:
    """High level controller exposing simple async methods."""

    def __init__(
        self,
        config_path: str | Path = "crypto_bot/config.yaml",
        trades_file: str | Path = LOG_DIR / "trades.csv",
        log_file: str | Path = LOG_DIR / "bot.log",
        wallet: "Wallet | PaperWallet | None" = None,
        paper_wallet: "PaperWallet | None" = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.trades_file = Path(trades_file)
        self.log_file = Path(log_file)
        self.config = self._load_config()
        self.rotator = PortfolioRotator()
        self.wallet = wallet or paper_wallet
        if self.wallet is None and self.config.get("execution_mode") == "dry_run":
            self.wallet = Wallet(
                self.config.get("start_balance", 1000.0),
                self.config.get("max_open_trades", 1),
                self.config.get("allow_short", False),
            )
        # Backwards compat – retain attribute name expected in some tests
        self.paper_wallet = self.wallet
        self.exchange, self.ws_client = get_exchange(self.config)
        self.proc: asyncio.subprocess.Process | None = None
        self.enabled: Dict[str, bool] = {
            "trend_bot": True,
            "grid_bot": True,
            "sniper_bot": True,
            "dex_scalper": True,
            "dca_bot": True,
            "mean_bot": True,
            "breakout_bot": True,
            "micro_scalp_bot": True,
            "bounce_scalper": True,
            "flash_crash_bot": True,

            "lstm_bot": True,
        }
        self.state = {
            "running": False,
            "mode": self.config.get("execution_mode", "dry_run"),
            "liquidate": False,
        }

    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        strat_dir = self.config_path.parent.parent / "config" / "strategies"
        trend_file = strat_dir / "trend_bot.yaml"
        if trend_file.exists():
            with open(trend_file) as sf:
                overrides = yaml.safe_load(sf) or {}
            trend_cfg = data.get("trend_bot", {})
            if isinstance(trend_cfg, dict):
                trend_cfg.update(overrides)
            else:
                trend_cfg = overrides
            data["trend_bot"] = trend_cfg

        if "symbol" in data:
            data["symbol"] = fix_symbol(data["symbol"])
        if "symbols" in data:
            data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
        return data

    async def start_trading(self) -> Dict[str, object]:
        """Launch ``crypto_bot.main`` as a subprocess if not already running."""
        if self.proc and self.proc.returncode is None:
            return {"running": True, "status": "already_running"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "crypto_bot.main",
            stdin=asyncio.subprocess.PIPE,
        )
        self.state["running"] = True
        return {"running": True, "status": "started"}

    async def send_command(self, cmd: str) -> None:
        """Send ``cmd`` followed by a newline to the subprocess stdin."""
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(cmd.encode() + b"\n")
            await self.proc.stdin.drain()

    async def stop_trading(self) -> Dict[str, object]:
        """Terminate the subprocess if running."""
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            await self.proc.wait()
            self.proc = None
        self.state["running"] = False
        return {"running": False, "status": "stopped"}

    async def get_status(self) -> Dict[str, object]:
        """Return current running state and enabled strategies."""
        running = self.proc is not None and self.proc.returncode is None
        self.state["running"] = running
        return {
            "running": running,
            "mode": self.state.get("mode"),
            "enabled_strategies": self.enabled.copy(),
        }

    async def list_strategies(self) -> List[str]:
        """Return names of available strategies."""
        return list(self.enabled.keys())

    async def toggle_strategy(self, name: str) -> Dict[str, object]:
        """Enable or disable ``name`` and return the new state."""
        if name not in self.enabled:
            raise ValueError(f"Unknown strategy: {name}")
        self.enabled[name] = not self.enabled[name]
        return {"strategy": name, "enabled": self.enabled[name]}

    async def list_positions(self) -> List[Dict]:
        """Return currently open positions parsed from the trade log."""
        return get_open_trades(self.trades_file)

    async def close_position(self, symbol: str, amount: float) -> Dict:
        """Submit a market order closing ``amount`` of ``symbol``."""
        order = await execute_trade_async(
            self.exchange,
            self.ws_client,
            symbol,
            "sell",
            amount,
            dry_run=self.config.get("execution_mode") == "dry_run",
            use_websocket=self.config.get("use_websocket", False),
            config=self.config,
        )
        if self.config.get("execution_mode") == "dry_run" and self.wallet:
            price = order.get("price") or 0.0
            if not price:
                try:
                    if asyncio.iscoroutinefunction(
                        getattr(self.exchange, "fetch_ticker", None)
                    ):
                        t = await self.exchange.fetch_ticker(symbol)
                    else:
                        t = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                    price = t.get("last") or t.get("bid") or t.get("ask") or 0.0
                except Exception:
                    price = 0.0
            if not price:
                entries = [p for p in get_open_trades(self.trades_file) if p.get("symbol") == symbol]
                if entries:
                    total = sum(float(p.get("amount", 0)) * float(p.get("price", 0)) for p in entries)
                    qty = sum(float(p.get("amount", 0)) for p in entries)
                    if qty > 0:
                        price = total / qty
            try:
                self.wallet.sell(symbol, amount, price)
                main.log_balance(self.wallet.total_balance)
            except Exception:
                pass

        balance = await main.fetch_and_log_balance(
            self.exchange, self.wallet, self.config
        )
        if isinstance(order, dict):
            order["balance"] = balance
        return order

    async def close_all_positions(self) -> Dict[str, str]:
        """Signal the trading bot to liquidate all open positions."""
        if self.proc and self.proc.returncode is None:
            await self.send_command("panic sell")
            return {"status": "command_sent"}
        self.state["liquidate_all"] = True
        return {"status": "liquidation_scheduled"}

    async def fetch_logs(self, lines: int = 20) -> List[str]:
        """Return the last ``lines`` from the bot log."""
        if not self.log_file.exists():
            return []
        data = self.log_file.read_text().splitlines()
        return data[-lines:]


    async def reload_config(self) -> Dict[str, object]:
        """Reload configuration from ``self.config_path``."""
        try:
            self.config = self._load_config()
            self.exchange, self.ws_client = get_exchange(self.config)
            self.state["mode"] = self.config.get("execution_mode", "dry_run")
            return {"status": "reloaded", "mode": self.state["mode"]}
        except Exception as exc:  # pragma: no cover - unexpected
            return {"status": "error", "error": str(exc)}


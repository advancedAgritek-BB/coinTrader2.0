from __future__ import annotations

"""Async wrapper controlling the trading bot."""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .portfolio_rotator import PortfolioRotator
from .utils.open_trades import get_open_trades
from .execution.cex_executor import get_exchange, execute_trade_async


class BotController:
    """High level controller exposing simple async methods."""

    def __init__(
        self,
        config_path: str | Path = "crypto_bot/config.yaml",
        trades_file: str | Path = "crypto_bot/logs/trades.csv",
        log_file: str | Path = "crypto_bot/logs/bot.log",
    ) -> None:
        self.config_path = Path(config_path)
        self.trades_file = Path(trades_file)
        self.log_file = Path(log_file)
        self.config = self._load_config()
        self.rotator = PortfolioRotator()
        self.exchange, self.ws_client = get_exchange(self.config)
        self.proc: asyncio.subprocess.Process | None = None
        self.enabled: Dict[str, bool] = {
            "trend_bot": True,
            "grid_bot": True,
            "sniper_bot": True,
            "dex_scalper": True,
            "mean_bot": True,
            "breakout_bot": True,
            "micro_scalp_bot": True,
            "bounce_scalper": True,
        }
        self.state = {
            "running": False,
            "mode": self.config.get("execution_mode", "dry_run"),
        }

    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    async def start_trading(self) -> Dict[str, object]:
        """Launch ``crypto_bot.main`` as a subprocess if not already running."""
        if self.proc and self.proc.returncode is None:
            return {"running": True, "status": "already_running"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "crypto_bot.main",
        )
        self.state["running"] = True
        return {"running": True, "status": "started"}

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
        return await execute_trade_async(
            self.exchange,
            self.ws_client,
            symbol,
            "sell",
            amount,
            dry_run=self.config.get("execution_mode") == "dry_run",
            use_websocket=self.config.get("use_websocket", False),
            config=self.config,
        )

    async def fetch_logs(self, lines: int = 20) -> List[str]:
        """Return the last ``lines`` from the bot log."""
        if not self.log_file.exists():
            return []
        data = self.log_file.read_text().splitlines()
        return data[-lines:]


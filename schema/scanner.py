from __future__ import annotations

import os
from pydantic import BaseModel, Field, validator


class ScannerConfig(BaseModel):
    """Configuration for market scanning."""

    scan_markets: bool = Field(True, description="Load all exchange pairs")
    symbols: list[str] | None = Field(default_factory=list, description="Symbols to trade")
    excluded_symbols: list[str] = Field(default_factory=list, description="Symbols to skip")
    exchange_market_types: list[str] = Field(default_factory=lambda: ["spot"], description="Market types")
    min_symbol_age_days: int = 0
    symbol_batch_size: int = 10
    scan_lookback_limit: int = 50
    cycle_lookback_limit: int | None = Field(
        default=None,
        description="Override per-cycle candle load (default min(150, timeframe_minutes * 2))",
    )
    max_spread_pct: float = 1.0

    class Config:
        extra = "allow"

    @validator("symbols", pre=True)
    def _default_symbols(cls, v):
        return v or []

    @validator("symbols")
    def _require_symbols_if_not_scanning(cls, v, values):
        if not values.get("scan_markets") and not v:
            raise ValueError("symbols must be provided when scan_markets is false")
        return v

    @validator("exchange_market_types", each_item=True)
    def _validate_market_type(cls, v):
        allowed = {"spot", "margin", "futures"}
        if v not in allowed:
            raise ValueError(f"invalid market type: {v}")
        return v

    @validator("symbol_batch_size", "scan_lookback_limit", "cycle_lookback_limit")
    def _positive_int(cls, v, field):
        if v is not None and v <= 0:
            raise ValueError(f"{field.name} must be > 0")
        return v


class SolanaScannerApiKeys(BaseModel):
    """API key configuration for Solana scanner."""

    moralis: str = Field(
        default_factory=lambda: os.getenv("MORALIS_KEY", "YOUR_KEY")
    )
    bitquery: str = Field(
        default_factory=lambda: os.getenv("BITQUERY_KEY", "YOUR_KEY")
    )
    lunarcrush_api_key: str = Field(
        default_factory=lambda: os.getenv("LUNARCRUSH_API_KEY", "YOUR_KEY")
    )
    raydium_api_key: str = Field(
        default_factory=lambda: os.getenv("RAYDIUM_API_KEY", "YOUR_KEY")
    )
    pump_fun_api_key: str = Field(
        default_factory=lambda: os.getenv("PUMP_FUN_API_KEY", "YOUR_KEY")
    )


class SolanaScannerConfig(BaseModel):
    """Configuration for scanning Solana tokens."""

    enabled: bool = Field(
        default=False, description="Enable Solana token scanner"
    )
    interval_minutes: float = Field(
        default=0.1,
        ge=0.05,
        description="Scan interval in minutes",
    )
    max_tokens_per_scan: int = Field(
        default=25,
        ge=1,
        description="Max tokens to process per scan",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum seconds to watch for new tokens",
    )
    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum iterations when scanning",
    )
    min_volume_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum USD volume for new tokens",
    )
    gecko_search: bool = Field(
        default=True, description="Enable GeckoTerminal search fallback"
    )
    # WebSocket / polling options
    helius_ws_url: str | None = Field(
        default=None,
        description=(
            "Helius WebSocket URL (e.g.,"
            " wss://mainnet.helius-rpc.com/?api-key=${HELIUS_KEY})"
        ),
    )
    raydium_program_id: str | None = Field(
        default="EhhTK0i58FmSPrbr30Y8wVDDDeWGPAHDq6oKEWSCPPu",
        description="Raydium program ID for subscription",
    )
    pump_fun_program_id: str | None = Field(
        default="EhhTK0i58FmSPrbr30Y8wVDDDeWGPAHDq6oKEWSCPPu",
        description="Pump.fun program ID for subscription",
    )
    use_ws: bool = Field(
        default=True, description="Prefer WebSocket over polling"
    )
    min_liquidity: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum liquidity for new pools (in SOL)",
    )
    api_keys: SolanaScannerApiKeys | None = None

    class Config:
        extra = "forbid"


class PythConfig(BaseModel):
    """Configuration for Pyth price feeds."""

    enabled: bool = False
    solana_endpoint: str = ""
    solana_ws_endpoint: str = ""
    program_id: str = ""

    class Config:
        extra = "forbid"


from __future__ import annotations

import os
from pydantic import BaseModel, Field, ValidationError, validator


class ScannerConfig(BaseModel):
    """Configuration for market scanning."""

    scan_markets: bool = Field(False, description="Load all exchange pairs")
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


class SolanaScannerConfig(BaseModel):
    """Configuration for scanning Solana tokens."""

    enabled: bool = False
    interval_minutes: int = 5
    api_keys: SolanaScannerApiKeys = Field(default_factory=SolanaScannerApiKeys)
    min_volume_usd: float = 0.0
    max_tokens_per_scan: int = 20
    gecko_search: bool = True

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


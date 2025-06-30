# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot. It can trade on centralized exchanges like Binance as well as on-chain DEXes on Solana.

Main features include:

- Regime detection using EMA, ADX, RSI and Bollinger Band width
- Strategy routing for trending, sideways, breakout or volatile markets
- Support for CEX strategies (trend and grid) and on-chain strategies (sniper and DEX scalper)
- Telegram notifications and optional Google Sheets logging
- Risk management with drawdown limits and volume/volatility filters
- Live trading or dry-run simulation
- Optional backtesting per regime

Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your credentials.

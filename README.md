# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot. It can trade on centralized exchanges like Coinbase Advanced Trade or Kraken as well as on-chain DEXes on Solana.

Main features include:

- Regime detection using EMA, ADX, RSI and Bollinger Band width
- Strategy routing for trending, sideways, breakout or volatile markets
- Support for CEX strategies (trend and grid) and on-chain strategies (sniper and DEX scalper)
- Telegram notifications and optional Google Sheets logging
- Risk management with drawdown limits and volume/volatility filters
- Live trading or dry-run simulation
- Optional backtesting per regime

Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your credentials.

## Exchange Setup for U.S. Users

1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. Fill out `crypto_bot/.env`:

   ```env
   EXCHANGE=coinbase  # or kraken
   API_KEY=your_key
   API_SECRET=your_secret
   API_PASSPHRASE=your_coinbase_passphrase_if_needed
   ```

3. In `crypto_bot/config.yaml` set:

   ```yaml
   exchange: coinbase  # Options: coinbase, kraken
   execution_mode: dry_run  # or live
   ```

Binance.US is not recommended because of API limitations.

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
- Experimental Kraken WebSocket support for real-time data and order updates

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

## Kraken WebSocket Usage

When using the `kraken` exchange the bot can obtain a WebSocket token via the
REST API and subscribe to real-time feeds. The optional module
`crypto_bot.data.kraken_ws` exposes a small `KrakenWebsocketClient` that handles
subscriptions to channels like `ohlc`, `book` and the private `executions`
stream. Trailing stop orders can also be submitted through the WebSocket
connection when enabled in your strategy.

WebSocket connectivity requires the `websockets` package which can be installed
with `pip install websockets`.

Binance.US is not recommended because of API limitations.

## Web UI

A small Flask web dashboard is included for running the bot and inspecting logs.
It features a responsive layout built with [Bootswatch](https://bootswatch.com/)
and provides separate pages for logs and trading statistics.

Start the UI with:

```bash
python -m frontend.app
```

Navigate to `http://localhost:5000` to start or stop the bot, watch the logs
refresh live and review the trade stats collected in
`crypto_bot/logs/strategy_stats.json`.

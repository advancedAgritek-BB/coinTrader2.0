# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot. It can trade on centralized exchanges like Coinbase Advanced Trade or Kraken as well as on-chain DEXes on Solana.

Main features include:

- Regime detection using EMA, ADX, RSI and Bollinger Band width
- Strategy routing for trending, sideways, breakout or volatile markets
- Support for CEX strategies (trend and grid) and on-chain strategies (sniper and DEX scalper)
- Telegram notifications and optional Google Sheets logging
- Risk management with drawdown limits and volume/volatility filters
- Live trading or dry-run simulation
- Backtesting with PnL, drawdown and Sharpe metrics
On-chain DEX execution on Solana now uses the Jupiter aggregator to submit real
transactions when not running in dry-run mode.


Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your credentials. The
script now asks for Coinbase and Kraken API keys (plus the Coinbase passphrase) and
stores your chosen exchange in `user_config.yaml`.

## Exchange Setup for U.S. Users

1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. Run `python crypto_bot/wallet_manager.py` and enter the keys when prompted.
3. Fill out `crypto_bot/.env`:

   ```env
   EXCHANGE=coinbase  # or kraken
   API_KEY=your_key
   API_SECRET=your_secret
   API_PASSPHRASE=your_coinbase_passphrase_if_needed
   ```


4. In `crypto_bot/config.yaml` set:
   For Kraken, optionally set tokens for the WebSocket API:

   ```env
   KRAKEN_WS_TOKEN=your_ws_token
   KRAKEN_API_TOKEN=your_api_token
   ```

   Generate `KRAKEN_WS_TOKEN` by calling Kraken's `GetWebSocketsToken` REST endpoint with your API credentials. The response contains a short-lived token used for authenticating WebSocket connections.

3. In `crypto_bot/config.yaml` set:

   ```yaml
   exchange: coinbase  # Options: coinbase, kraken
   execution_mode: dry_run  # or live
   use_websocket: true      # enable when trading on Kraken via WebSocket
   ```

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

### Backtesting

The `backtest` function in `crypto_bot.backtest.backtest_runner` can evaluate
different stop‑loss and take‑profit percentages and reports the PnL,
maximum drawdown and Sharpe ratio for each combination.

```python
from crypto_bot.backtest import backtest_runner

results = backtest_runner.backtest(
    'BTC/USDT', '1h', since=0,
    stop_loss_range=[0.01, 0.02],
    take_profit_range=[0.02, 0.04],
)
print(results.head())  # best combo appears first
```

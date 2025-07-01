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


Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your user preferences.
API keys are read from environment variables when present. Place them in
`crypto_bot/.env` or export them in your shell. Sensitive values will not be saved to
`user_config.yaml` unless you set a `FERNET_KEY`, in which case they are encrypted
before being written.

## Exchange Setup for U.S. Users

1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. Run `python crypto_bot/wallet_manager.py` to generate `user_config.yaml`. Any
   credentials found in the environment will be used automatically.
3. Fill out `crypto_bot/.env` with your API keys and optional `FERNET_KEY`:

   ```env
   EXCHANGE=coinbase  # or kraken
   API_KEY=your_key
   API_SECRET=your_secret
   API_PASSPHRASE=your_coinbase_passphrase_if_needed
   FERNET_KEY=optional_key_for_encryption
   ```


4. In `crypto_bot/config.yaml` set:
   For Kraken, optionally set tokens for the WebSocket API:

   ```env
   KRAKEN_WS_TOKEN=your_ws_token
   KRAKEN_API_TOKEN=your_api_token
   ```

   Generate `KRAKEN_WS_TOKEN` by calling Kraken's `GetWebSocketsToken` REST endpoint with your API credentials. The response contains a short-lived token used for authenticating WebSocket connections.

5. In `crypto_bot/config.yaml` set:

    ```yaml
    exchange: coinbase  # Options: coinbase, kraken
    execution_mode: dry_run  # or live
    use_websocket: true      # enable when trading on Kraken via WebSocket
    ```

Additional execution flags:

```yaml
liquidity_check: true        # verify order book liquidity before placing orders
liquidity_depth: 10          # order book depth levels to inspect
twap_enabled: false          # split large orders into slices
twap_slices: 4               # number of slices when TWAP is enabled
twap_interval_seconds: 10    # delay between TWAP slices
```

When `use_websocket` is enabled the bot relies on `ccxt.pro` for realtime
streaming data. Install it alongside the other requirements or disable
websockets if you do not have access to `ccxt.pro`.

Binance.US is not recommended because of API limitations.

## Web UI

A small Flask web dashboard is included for running the bot and inspecting logs.
It features a responsive layout built with [Bootswatch](https://bootswatch.com/)
and provides separate pages for logs and trading statistics. A background
watchdog thread now monitors the trading bot and automatically restarts it if
the process exits unexpectedly.

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
`crypto_bot/logs/strategy_stats.json`. The home page indicates whether the bot
is running so you can quickly see if it has stopped.

# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot capable of operating on centralized exchanges like Coinbase Advanced Trade or Kraken and on Solana DEXes via the Jupiter aggregator.

## Features

* Regime detection using EMA, ADX, RSI and Bollinger Band width
* Strategy router that picks the best approach for trending, sideways, breakout or volatile markets
* Trend and grid bots for CEXs plus sniper and DEX scalper strategies on Solana
* Portfolio rotation and auto optimizer utilities
* Risk management with drawdown limits, cooldown management and volume/volatility filters
* Telegram notifications and optional Google Sheets logging
* Capital tracker, sentiment filter and tax logger helpers
* Solana mempool monitor to avoid swaps when fees spike
* Paper trading wallet for dry-run simulation
* Live trading or dry-run simulation
* Web dashboard with watchdog thread and realtime log view
* Backtesting module with PnL, drawdown and Sharpe metrics

On-chain DEX execution submits real transactions when not running in dry-run mode.

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run `python crypto_bot/wallet_manager.py` to create `user_config.yaml` and enter your API credentials.
3. Adjust `crypto_bot/config.yaml` to select the exchange and execution mode.
4. Start the trading bot:
   ```bash
   python -m crypto_bot.main
   ```
   When dry-run mode is selected you will be prompted for the starting USDT balance.
   Or launch the web dashboard with:
   ```bash
   python -m frontend.app
   ```

Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your credentials. The
script now prompts only for the API keys required by the exchange you select and
stores your choice in `user_config.yaml`. **Environment variables always
override the values in this file**, so you can place secrets in `crypto_bot/.env` or
export them before running the bot.

To pull secrets from a provider such as AWS Secrets Manager or Hashicorp Vault,
set the environment variables `SECRETS_PROVIDER` (`aws` or `vault`) and
`SECRETS_PATH` to the secret name/path. When configured the credentials will be
loaded automatically.
Edit `crypto_bot/config.yaml` and run `wallet_manager.py` to set up your user preferences.
API keys are read from environment variables when present. Place them in
`crypto_bot/.env` or export them in your shell. Sensitive values will not be saved to
`user_config.yaml` unless you set a `FERNET_KEY`, in which case they are encrypted
before being written.

## Exchange Setup for U.S. Users

1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. Run `python crypto_bot/wallet_manager.py` to generate `user_config.yaml`. Any
   credentials found in the environment will be used automatically.
3. Fill out `crypto_bot/.env` with your API keys and optional `FERNET_KEY`.
   Environment variables take precedence over values stored in
   `user_config.yaml`. If you prefer to enter credentials interactively,
   leave the entries commented out.

   ```env
   # EXCHANGE=coinbase  # or kraken
   # API_KEY=your_key
   # API_SECRET=your_secret
   # API_PASSPHRASE=your_coinbase_passphrase_if_needed
   # FERNET_KEY=optional_key_for_encryption
   ```

### Twitter Sentiment API

Set `TWITTER_SENTIMENT_URL` to the endpoint for the sentiment service used by
`sentiment_filter.py`. If this variable is not provided, the bot defaults to the
placeholder `https://api.example.com/twitter-sentiment`, so sentiment fetches
will fail until a real URL is supplied.

### Funding Rate API

Set `FUNDING_RATE_URL` to the endpoint used by `volatility_filter.py` when
fetching perpetual funding rates. Without this variable the bot falls back to
the placeholder `https://funding.example.com` and will log errors until a real
URL is supplied.

For Kraken, set:

```env
FUNDING_RATE_URL=https://api.kraken.com/0/public/Ticker
```

`volatility_filter.py` will append `?pair=SYMBOL` to this URL when requesting
funding information.


4. In `crypto_bot/config.yaml` set:
   For Kraken, optionally set tokens for the WebSocket API:

   ```env
   KRAKEN_WS_TOKEN=your_ws_token
   KRAKEN_API_TOKEN=your_api_token
   ```

Generate `KRAKEN_WS_TOKEN` by calling Kraken's `GetWebSocketsToken` REST endpoint with your API credentials. The response contains a short-lived token used for authenticating WebSocket connections. The WebSocket client connects to the `/v2` URLs (`wss://ws.kraken.com/v2` and `wss://ws-auth.kraken.com/v2`), so the token is required for trading. A helper is provided in `crypto_bot.utils`:

```python
from crypto_bot.utils import get_ws_token
token = get_ws_token(API_KEY, API_SECRET, "123456")
```

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


The Kraken WebSocket client automatically reconnects if the connection drops and
resubscribes to any previously requested channels.  Trading commands use the new
`/v2` naming scheme such as `add_order`, `cancel_order`, `cancel_all_orders` and
`open_orders`.  Refer to Kraken's v2 WebSocket documentation for a full list:
<https://docs.kraken.com/websockets-v2/#tag/Trading>.

Binance.US is not recommended because of API limitations.

### Automatic Market Scanning

When `scan_markets` is set to `true` and the `symbols` list is empty, the bot
loads all active Kraken trading pairs at startup. Pairs listed under
`excluded_symbols` are skipped. Disable this behaviour by setting
`scan_markets` to `false`.

```yaml
scan_markets: true
symbols: []            # automatically populated
excluded_symbols: [ETH/USD]
```

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
`crypto_bot/logs/strategy_stats.json`. When the bot is stopped a form
lets you select the execution mode (dry run or live) before launching.

## Solana Mempool Monitoring

The bot can monitor Solana priority fees to avoid swaps when congestion
is high. Enable the monitor in `crypto_bot/config.yaml`:

```yaml
mempool_monitor:
  enabled: true
  suspicious_fee_threshold: 100
  action: pause  # or reprice
  reprice_multiplier: 1.05
```

When enabled, `execute_swap` checks the current priority fee and pauses
or adjusts the trade according to the selected action.
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

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
* Trade history page highlighting buys in green and sells in red
* Backtesting module with PnL, drawdown and Sharpe metrics

On-chain DEX execution submits real transactions when not running in dry-run mode.

## Regime Classifier

The bot selects a strategy by first classifying the current market regime. The
`classify_regime` function computes EMA, ADX, RSI and Bollinger Band width to
label conditions as `trending`, `sideways`, `breakout`, `mean-reverting` or
`volatile`. At least **20** OHLCV entries are required for these indicators to
be calculated reliably. When fewer rows are available the function returns
`"unknown"` so the router can avoid making a poor decision.

### Optional ML Fallback

Set `use_ml_regime_classifier` to `true` in `crypto_bot/config.yaml` to fall
back to a machine learning model whenever the indicator rules return
`"unknown"`.  A small fallback model is bundled directly in
`crypto_bot.regime.model_data` as a base64 string and loaded automatically.
By default the ML model only runs when at least **20** candles are available
(tunable via `ml_min_bars`).  You can replace that module with your own
encoded model if desired.

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The optional `rich` package is included and provides colorized
   console output when viewing live positions.
2. Run `python crypto_bot/wallet_manager.py` to create `user_config.yaml` and enter your API credentials.
3. Adjust `crypto_bot/config.yaml` to select the exchange and execution mode.
4. Start the trading bot:
   ```bash
   python -m crypto_bot.main
   ```
   When dry-run mode is selected you will be prompted for the starting USDT balance.
   The console now refreshes with your wallet balance and any active
   trades in real time. Profitable positions are shown in green while
   losing ones appear in red. A second line now lists each open trade with
   its running profit or loss.
   The program prints "Bot running..." before the [Monitor] lines.
   Type `start`, `stop` or `quit` in the terminal to control the bot.
   Or launch the web dashboard with:
   ```bash
   python -m frontend.app
   ```

Run `wallet_manager.py` to create `user_config.yaml` and enter your exchange credentials. Values from `crypto_bot/.env` override those stored in `user_config.yaml`. Setting `SECRETS_PROVIDER` (`aws` or `vault`) with `SECRETS_PATH` loads credentials automatically. Provide a `FERNET_KEY` to encrypt sensitive values in `user_config.yaml`.

## Configuration Files

The bot looks in two locations for settings:

1. **`crypto_bot/.env`** – store API keys and other environment variables here. These values override entries loaded by `wallet_manager.py`.
2. **`crypto_bot/config.yaml`** – general runtime options controlling strategy behaviour and notifications.

### Environment Variables

Create `crypto_bot/.env` and fill in your secrets. Example:

```env
EXCHANGE=coinbase              # or kraken
API_KEY=your_key
API_SECRET=your_secret
API_PASSPHRASE=your_coinbase_passphrase_if_needed
FERNET_KEY=optional_key_for_encryption
KRAKEN_WS_TOKEN=your_ws_token          # optional for Kraken
KRAKEN_API_TOKEN=your_api_token        # optional for Kraken
TELEGRAM_TOKEN=your_telegram_token
GOOGLE_CRED_JSON=path_to_google_credentials.json
TWITTER_SENTIMENT_URL=https://api.example.com/twitter-sentiment
FUNDING_RATE_URL=https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=
SECRETS_PROVIDER=aws                     # optional
SECRETS_PATH=/path/to/secret
```

### YAML Configuration

Edit `crypto_bot/config.yaml` to adjust trading behaviour. Key settings include:

```yaml
exchange: coinbase       # coinbase or kraken
execution_mode: dry_run  # or live
use_websocket: true
telegram:
  token: your_telegram_token
  chat_id: your_chat_id
  trade_updates: true
mempool_monitor:
  enabled: false
  suspicious_fee_threshold: 100
  action: pause
  reprice_multiplier: 1.05
```

## Configuration Options

The `crypto_bot/config.yaml` file holds the runtime settings for the bot. Below is a high level summary of what each option controls.

### Exchange and Execution
* **exchange** – target CEX (`coinbase` or `kraken`).
* **execution_mode** – choose `dry_run` for simulation or `live` for real orders.
* **use_websocket** – enable WebSocket data via `ccxt.pro`.
* **force_websocket_history** – disable REST fallbacks when streaming.
* **exchange_market_types** – market types to trade (spot, margin, futures).
* **preferred_chain** – chain used for on-chain swaps (e.g. `solana`).
* **wallet_address** – destination wallet for DEX trades.
* **symbol**/**symbols** – pairs to trade when not scanning automatically.
* **scan_markets** – load all exchange pairs when `symbols` is empty.
* **excluded_symbols** – markets to skip during scanning.

### Market Scanning
* **symbol_batch_size** – number of symbols processed each cycle.
* **symbol_refresh_minutes** – minutes before the symbol queue is refreshed.
* **symbol_filter** – volume and spread limits for candidate pairs.
* **symbol_score_weights** – weights for volume, spread, change and age.
* **min_symbol_age_days** – skip newly listed pairs.
* **min_symbol_score** – minimum score required for trading.
* **top_n_symbols** – maximum number of active markets.
* **max_age_days**, **max_change_pct**, **max_spread_pct**, **max_latency_ms**, **max_vol** – additional scanning limits.

### Risk Parameters
* **risk** – default stop loss, take profit and drawdown limits.
* **trade_size_pct** – percent of capital used per trade.
* **max_slippage_pct** – slippage tolerance for orders.
* **liquidity_check**/**liquidity_depth** – verify order book depth.
* **volatility_filter** – ATR and funding rate thresholds.
* **sentiment_filter** – fear & greed plus sentiment checks.
* **sl_pct**/**tp_pct** – defaults for Solana scalper strategies.
* **mempool_monitor** – pause or reprice when Solana fees spike.
* **min_cooldown** – minimum minutes between trades.
* **cycle_bias** – optional on-chain metrics to bias trades.

### Strategy and Signals
* **strategy_allocation** – capital split across strategies.
* **strategy_evaluation_mode** – how the router chooses a strategy.
* **voting_strategies**/**min_agreeing_votes** – strategies used for the voting router.
* **exit_strategy** – partial profit taking and trailing stop logic.
* **micro_scalp** – EMA and volume settings for the scalp bot.
* **ml_signal_model**/**signal_weight_optimizer** – blend strategy scores with machine-learning predictions.
* **signal_threshold**, **min_confidence_score**, **min_consistent_agreement** – thresholds for entering a trade.
* **regime_timeframes**/**regime_return_period** – windows used for regime detection.
* **twap_enabled**, **twap_slices**, **twap_interval_seconds** – settings for time-weighted order execution.
* **optimization** – periodic parameter optimisation.
* **portfolio_rotation** – rotate holdings based on scoring metrics.
* **meta_selector**/**rl_selector** – experimental strategy routers.
* **mode** – `auto` or `manual` evaluation of strategies.

### Data and Logging
* **timeframe**, **timeframes**, **scalp_timeframe** – candle intervals used for analysis.
* **ohlcv_snapshot_frequency_minutes**/**ohlcv_snapshot_limit** – OHLCV caching options.
* **loop_interval_minutes** – delay between trading cycles.
* **log_to_google** – export trades to Google Sheets.
* **telegram** – bot token, chat ID and trade notifications.
* **tax_tracking** – CSV export of executed trades.
* **metrics_enabled**, **metrics_backend**, **metrics_output_file** – cycle metrics output.
* **testing_mode** – indicates a sandbox environment.
## Exchange Setup for U.S. Users


1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. Run `python crypto_bot/wallet_manager.py` to generate `user_config.yaml`. Any
   credentials found in the environment will be used automatically.
3. Place your API keys in `crypto_bot/.env` as shown in the configuration
   section above. Environment variables take precedence over values stored in
   `user_config.yaml`.
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

### Telegram Setup

1. Open `crypto_bot/config.yaml` and fill in the `telegram` section:

   ```yaml
   telegram:
     token: your_telegram_token
     chat_id: your_chat_id
     trade_updates: true
   ```

   The bot reads the chat ID and token from `config.yaml` (not
   `user_config.yaml`). Set `trade_updates` to `false` to disable trade entry
   and exit messages.
     trade_updates: true  # set false to disable trade notifications
   ```

   The bot reads these values only from `config.yaml`. Disable
   `trade_updates` if you don't want trade entry and exit messages.
2. Send `/start` to your bot so it can message you. Use `/menu` at any time to
   open a list of buttons—**Start**, **Stop**, **Status**, **Log**, **Rotate
   Now** and **Toggle Mode**—for quick interaction.
3. If you see `Failed to send message: Not Found` in the logs, the chat ID or
   token is likely incorrect or the bot lacks permission to message the chat.
   Double-check the values in `config.yaml` and ensure you've started a
   conversation with your bot.

### Twitter Sentiment API

Add `TWITTER_SENTIMENT_URL` to `crypto_bot/.env` to point at the sentiment
service used by `sentiment_filter.py`. If this variable is not provided, the bot
defaults to the placeholder `https://api.example.com/twitter-sentiment`, so
sentiment fetches will fail until a real URL is supplied.

### Funding Rate API

Add `FUNDING_RATE_URL` to `crypto_bot/.env` to specify the endpoint used by
`volatility_filter.py` when fetching perpetual funding rates. Without this
variable the bot falls back to the placeholder `https://funding.example.com`
and will log errors until a real URL is supplied.

For Kraken, add the following line to `crypto_bot/.env`:

```env
FUNDING_RATE_URL=https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=
```

`volatility_filter.py` will append the instrument symbol directly to this URL
when requesting funding information.


4. In `crypto_bot/config.yaml` set:

   ```yaml
   exchange: coinbase  # Options: coinbase, kraken
   execution_mode: dry_run  # or live
   use_websocket: true      # enable when trading on Kraken via WebSocket
   ```

   For Kraken, optionally place WebSocket tokens in `crypto_bot/.env`:

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
timeframe: 1h                # candles for regime detection
scalp_timeframe: 1m          # candles for micro_scalp/bounce_scalper
loop_interval_minutes: 5     # wait time between trading cycles
force_websocket_history: false  # set true to disable REST fallback
ohlcv_timeout: 10            # request timeout for OHLCV fetches
max_concurrent_ohlcv: 20     # limit simultaneous OHLCV fetches
metrics:
  enabled: true              # write cycle statistics to metrics.csv
  file: crypto_bot/logs/metrics.csv
```

`loop_interval_minutes` determines how long the bot sleeps between each
evaluation cycle, giving the market time to evolve before scanning again.
`max_concurrent_ohlcv` caps how many OHLCV requests run in parallel when
`update_ohlcv_cache` gathers new candles. The new `ohlcv_timeout` option
controls the timeout for each fetch call. If you still encounter timeouts after
raising this value, try lowering `max_concurrent_ohlcv` to reduce pressure on
the exchange API.
The `metrics` section enables recording of cycle summaries to the specified CSV
file for later analysis.
`scalp_timeframe` sets the candle interval specifically for the micro_scalp
and bounce_scalper strategies while `timeframe` covers all other analysis.

When `use_websocket` is enabled the bot relies on `ccxt.pro` for realtime
streaming data. Install it alongside the other requirements or disable
websockets if you do not have access to `ccxt.pro`.
When OHLCV streaming returns fewer candles than requested the bot fills the gap using REST to ensure indicators have enough history. Disable this fallback by setting `force_websocket_history` to `true`.

Example usage for Kraken WebSockets:

```python
from crypto_bot.execution.kraken_ws import KrakenWSClient

ws = KrakenWSClient()
ws.subscribe_orders(["BTC/USD"])  # open_orders channel
ws.add_order(["BTC/USD"], "buy", 0.01)
ws.cancel_order("OABCDEF", ["BTC/USD"])
```

The Kraken WebSocket client automatically reconnects if the connection drops and
resubscribes to any previously requested channels.  Trading commands use the new
`/v2` naming scheme such as `add_order`, `cancel_order`, `cancel_all_orders` and
`open_orders`.  Refer to Kraken's v2 WebSocket documentation for a full list:
<https://docs.kraken.com/websockets-v2/#tag/Trading>.

Example usage:

```python
from crypto_bot.execution.kraken_ws import KrakenWSClient

client = KrakenWSClient(ws_token="your_ws_token")
client.add_order("BTC/USD", "buy", 0.1)
client.cancel_order("TXID123")
client.cancel_all_orders()
client.open_orders()
```

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
exchange_market_types: ["spot"]  # options: spot, margin, futures
min_symbol_age_days: 10          # skip pairs with less history
symbol_batch_size: 10            # symbols processed per cycle
max_spread_pct: 1.0              # skip pairs with wider spreads
```

`exchange_market_types` filters the discovered pairs by market class. The bot
also skips newly listed pairs using `min_symbol_age_days`.
Symbols are queued by score using a priority deque and processed in
batches controlled by `symbol_batch_size`. When the queue drops below this
size it is automatically refilled with the highest scoring symbols.
Candidates are stored in a priority queue sorted by their score so the highest
quality markets are scanned first. Each cycle pulls the next `symbol_batch_size`
symbols from this queue and refills it when empty.

OHLCV data for these symbols is now fetched concurrently using
`load_ohlcv_parallel`, greatly reducing the time needed to evaluate
large symbol lists.

Each candidate pair is also assigned a score based on volume, recent price
change, bid/ask spread, age and API latency. The weights and limits for this
calculation can be tuned via `symbol_score_weights`, `max_vol`,
`max_change_pct`, `max_spread_pct`, `max_age_days` and `max_latency_ms` in
`config.yaml`. Only symbols with a score above `min_symbol_score` are included
in trading rotations.
## Symbol Filtering

The bot evaluates each candidate pair using Kraken ticker data. By
setting options under `symbol_filter` you can weed out illiquid or
undesirable markets before strategies run:

```yaml
symbol_filter:
  min_volume_usd: 50000         # minimum 24h volume in USD
  change_pct_percentile: 70     # require 24h change in the top 30%
  max_spread_pct: 0.5           # skip pairs with wide spreads
  correlation_window: 30        # days of history for correlation
  max_correlation: 0.9          # drop pairs above this threshold
```

Pairs passing these checks are then scored with `analyze_symbol` which
computes a strategy confidence score. Only the highest scoring symbols
are traded each cycle.

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
refresh live and review the trade statistics stored in
`crypto_bot/logs/strategy_stats.json` and the detailed performance records in
`crypto_bot/logs/strategy_performance.json`. When the bot is stopped a form
lets you select the execution mode (dry run or live) before launching.

## Log Files

All runtime information is written under `crypto_bot/logs`. Important files
include:

- `bot.log` – main log file containing startup events, strategy choices and all
  decision messages.
- `trades.csv` – CSV export of every executed trade used by the dashboard and
  backtester. Stop orders are logged here as well with an `is_stop` flag so they
  can be filtered out from performance calculations.
- `strategy_stats.json` – summary statistics of win rate, PnL and other metrics.
- `strategy_performance.json` – list of individual trades grouped by regime and
  strategy with fields like `pnl` and timestamps.
- `metrics.csv` – per cycle summary showing how many pairs were scanned,
  how many signals fired and how many trades executed.

### Statistics File Structure

`strategy_performance.json` stores raw trade records nested by market regime and
strategy. Example snippet:

```json
{
  "trending": {
    "trend_bot": [
      {
        "symbol": "BTC/USDT",
        "pnl": 1.2,
        "entry_time": "2024-01-01T00:00:00Z",
        "exit_time": "2024-01-01T02:00:00Z"
      }
    ]
  }
}
```

`strategy_stats.json` contains aggregated statistics per strategy such as win
rate and average PnL:

```json
{
  "trend_bot": {
    "trades": 10,
    "win_rate": 0.6,
    "avg_win": 1.2,
    "avg_loss": -0.8
  }
}
```

Other helpers create logs like `execution.log` in the same directory when enabled. Decision details are consolidated in `bot.log`, letting you follow the router and risk manager actions in one place. Example snippet:

```text
2024-02-12 15:04:01 - INFO - Starting bot
2024-02-12 15:04:02 - INFO - Strategy router selected grid_bot for BTC/USDT
2024-02-12 15:04:10 - INFO - Placing buy order amount 0.1 price 22000
2024-02-12 15:04:15 - INFO - Decision: take profit triggered at 22400
```

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

## ML Signal Model

Strategy scores can be blended with predictions from an optional machine
learning model. Configure the feature in `crypto_bot/config.yaml`:

```yaml
ml_signal_model:
  enabled: false        # enable ML scoring
  weight: 0.5           # blend ratio between strategy and ML scores
```

When enabled, `evaluate` computes `(score * (1 - weight)) + (ml_score * weight)`
and caps the result between 0 and 1.

## Development Setup

1. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

2. Run the test suite to verify your environment:

```bash
pytest -q
```

## Troubleshooting

High `max_concurrent_ohlcv` values combined with short `ohlcv_timeout`
settings can overload the exchange and lead to failed candle fetches.
Increase `ohlcv_timeout` to give each request more time and lower
`max_concurrent_ohlcv` if errors continue.

This project is provided for educational purposes only. Use it at your own risk, and remember that nothing here constitutes financial advice.


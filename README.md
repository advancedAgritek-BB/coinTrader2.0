# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot capable of operating on centralized exchanges like Coinbase Advanced Trade or Kraken and on Solana DEXes via the Jupiter aggregator.

## Features

* Regime detection using EMA, ADX, RSI and Bollinger Band width
* Strategy router that picks the best approach for trending, sideways, breakout or volatile markets
* Fast-path dispatcher that jumps straight to the breakout or trend bot on strong signals
* Trend and grid bots for CEXs plus sniper and DEX scalper strategies on Solana
* Portfolio rotation and auto optimizer utilities
* Risk management with drawdown limits, cooldown management and volume/volatility filters
* Telegram notifications and optional Google Sheets logging
* Interactive Telegram menu with buttons for start/stop, PnL stats, trade history and config editing
* Balance change alerts when USDT funds move
* Capital tracker, sentiment filter and tax logger helpers
* Solana mempool monitor to avoid swaps when fees spike
* Paper trading wallet for dry-run simulation
* Live trading or dry-run simulation
* Web dashboard with watchdog thread and realtime log view
* Trade history page highlighting buys in green and sells in red
* Backtesting module with PnL, drawdown and Sharpe metrics
* Utility functions automatically handle synchronous or asynchronous exchange clients

On-chain DEX execution submits real transactions when not running in dry-run mode.

## Regime Classifier

The bot selects a strategy by first classifying the current market regime. The
`classify_regime` function computes EMA, ADX, RSI and Bollinger Band width to
label conditions as `trending`, `sideways`, `breakout`, `mean-reverting` or
`volatile`. At least **200** candles are required for these indicators to
be calculated reliably. When fewer rows are available the function returns
`"unknown"` so the router can avoid making a poor decision. Strategies may
operate on different candle intervals, so the loader keeps a multi‑timeframe
cache populated for each pair. The `timeframes` list in
`crypto_bot/config.yaml` defines which intervals are stored and reused across
the various bots.

### Optional ML Fallback

Set `use_ml_regime_classifier` to `true` in `crypto_bot/config.yaml` to fall
back to a machine learning model whenever the indicator rules return
`"unknown"`.  A small gradient boosting model trained with LightGBM is bundled
directly in `crypto_bot.regime.model_data` as a base64 string and loaded
automatically.
`use_ml_regime_classifier` is enabled by default in
`crypto_bot/regime/regime_config.yaml`, so the router falls back to a small
machine learning model whenever the indicator rules return `"unknown"`.
The EMA windows have been shortened to **8** and **21** and the ADX threshold
lowered to switch regimes more quickly. The fallback model is bundled in
`crypto_bot.regime.model_data` as a base64 string and loaded automatically.
By default the ML model only runs when at least **200** candles are available
(tunable via `ml_min_bars`). You can replace that module with your own encoded
model if desired.
When enough history is present the ML probabilities are blended with the
rule-based result using `ml_blend_weight` from `regime_config.yaml`.

The regime configuration exposes additional tuning parameters:

* **adx_trending_min** – ADX threshold for the trending regime.
* **breakout_volume_mult** – volume multiplier for breakout detection.
* **score_weights** – weighting factors for regime probabilities when patterns
  are detected.
* **pattern_min_conf** – minimum pattern confidence required to apply a score
  weight.
* **ml_blend_weight** – blend ratio for combining ML and indicator scores.
* **bull_fng** – Fear & Greed index level considered bullish.
* **atr_baseline** – ATR level corresponding to a 1× score factor.

## Fast-Path Checks

The router performs quick checks for exceptionally strong setups before running
the full regime classifier. When the Bollinger Band width over a 20 candle
window drops below **0.05** and volume is more than **5×** the average,
`breakout_bot` is called immediately. If the ADX from the same window exceeds
**35**, the router dispatches straight to `trend_bot`. These defaults live under
`strategy_router.fast_path` in `crypto_bot/config.yaml` and can be tuned as
needed.

## Quick Start

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The optional `rich` package is included and provides colorized
   console output when viewing live positions.
   Exchange connectivity relies on [ccxt](https://github.com/ccxt/ccxt) which is installed with these requirements. Make sure the `ccxt` package is available when running the trading bot.
2. Run `python crypto_bot/wallet_manager.py` to create `user_config.yaml` and enter your API credentials.
3. Adjust `crypto_bot/config.yaml` to select the exchange and execution mode.
4. Start the trading bot:
   ```bash
   python -m crypto_bot.main
   ```
   When dry-run mode is selected you will be prompted for the starting USDT balance.
   The console now refreshes with your wallet balance and any active
   trades in real time. Profitable positions are shown in green while
   losing ones appear in red. The monitor lists open trades on a single
   line formatted as `Symbol -- entry -- unrealized PnL`.
   The program prints "Bot running..." before the [Monitor] lines.
   Before trading begins the bot performs a full market scan to populate
   its caches. Progress is logged and, when `telegram.status_updates` is
   enabled, sent to your Telegram chat.
   Type `start`, `stop`, `reload` or `quit` in the terminal to control the bot.
   Or launch the web dashboard with:
   ```bash
   python -m frontend.app
   ```
5. Run the meme-wave sniper separately with Raydium v3 integration.
   Profits are automatically converted to BTC. Set `SOLANA_PRIVATE_KEY` and
   `HELIUS_KEY` or provide a custom `SOLANA_RPC_URL` before launching:
   ```bash
   python -m crypto_bot.solana.runner
   ```

6. Edit `crypto_bot/config.yaml` and reload the settings without restarting the
   bot:

   ```yaml
   risk:
     trade_size_pct: 1.5  # new value
   ```

   Save the file and type `reload` in the console or send `/reload` via Telegram
   to apply the changes immediately.

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
TELE_CHAT_ADMINS=123456,789012         # optional comma separated admin IDs
TELE_CHAT_ADMINS=12345,67890          # comma-separated chat IDs
GOOGLE_CRED_JSON=path_to_google_credentials.json
TWITTER_SENTIMENT_URL=https://api.example.com/twitter-sentiment
FUNDING_RATE_URL=https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=
SECRETS_PROVIDER=aws                     # optional
SECRETS_PATH=/path/to/secret
SOLANA_PRIVATE_KEY="[1,2,3,...]"       # required for Solana trades
# defaults to https://mainnet.helius-rpc.com/?api-key=${HELIUS_KEY}
SOLANA_RPC_URL=https://devnet.solana.com  # optional custom endpoint
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com  # optional
# SOLANA_RPC_URL=https://api.devnet.solana.com      # devnet example
HELIUS_KEY=your_helius_api_key          # optional, for Helius RPC endpoints
MORALIS_KEY=your_moralis_api_key       # optional, for Solana scanner
BITQUERY_KEY=your_bitquery_api_key     # optional, for Solana scanner
```

`TELE_CHAT_ADMINS` lets the Telegram bot accept commands from multiple
admin chats. Omit it to restrict control to the single `chat_id` in the
configuration file.

### Solana Setup

Example RPC URLs:

```env
# Mainnet
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
# Devnet
# SOLANA_RPC_URL=https://api.devnet.solana.com
```

When using [Helius](https://www.helius.xyz/) endpoints, append `?api-key=${HELIUS_KEY}` to the URL:

```env
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_KEY}
```

You can generate a key and enable advanced features like **ShredStream** and **LaserStream** from the [Helius dashboard](https://dashboard.helius.xyz/). These streams can be configured directly in the bot's web dashboard.
Install the `pythclient` package to fetch oracle prices:

```bash
pip install pythclient
```

Add a `pyth` section to `crypto_bot/config.yaml`:

```yaml
pyth:
  enabled: false
  solana_endpoint: https://api.mainnet-beta.solana.com
  solana_ws_endpoint: wss://api.mainnet-beta.solana.com
  program_id: FsJ3A3u2vn5cTVofAjvy6qM3HrjTXg5Gs1Y8D6fCt3m
```
These keys are required to connect to Pyth and can be adjusted for your
environment.



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
  status_updates: true
  balance_updates: false
  mempool_monitor:
    enabled: false
    suspicious_fee_threshold: 100
    action: pause
    reprice_multiplier: 1.05
  bandit:
    enabled: false
    alpha0: 1
    beta0: 1
    explore_pct: 0.05
```

## Configuration Options

The `crypto_bot/config.yaml` file holds the runtime settings for the bot. Below is a high level summary of what each option controls.

### Exchange and Execution
* **exchange** – target CEX (`coinbase` or `kraken`).
* **execution_mode** – choose `dry_run` for simulation or `live` for real orders.
  Paper trading defaults to long-only on spot exchanges.
* **use_websocket** – enable WebSocket data via `ccxt.pro`.
* **force_websocket_history** – disable REST fallbacks when streaming (default: true).
* **max_ws_limit** – skip WebSocket OHLCV when `limit` exceeds this value.
* **exchange_market_types** – market types to trade (spot, margin, futures).
* **preferred_chain** – chain used for on-chain swaps (e.g. `solana`).
* **wallet_address** – destination wallet for DEX trades.
* **solana_slippage_bps** – slippage tolerance for on-chain conversions.
* **symbol**/**symbols** – pairs to trade when not scanning automatically.
* **scan_markets** – load all exchange pairs when `symbols` is empty.
* **scan_in_background** – start the initial scan in the background so trading can begin immediately.
* **excluded_symbols** – markets to skip during scanning.
* **solana_symbols** – base tokens traded on Solana; each is appended with `/USDC`.
* **allow_short** – enable short selling. Set to `true` only when your exchange account supports short selling.

### Market Scanning
* **symbol_batch_size** – number of symbols processed each cycle.
  The same batch size controls the initial market scan at startup where
  progress is logged after each batch.
* **scan_lookback_limit** – candles of history loaded during the initial scan (default `700`).
  The caches store at least this many bars per timeframe before strategies run.
  Initial history is retrieved via REST with up to 700 candles per timeframe.
* **cycle_lookback_limit** – candles fetched each cycle. Defaults to `150`.
* **adaptive_scan.enabled** – turn on dynamic sizing.
* **adaptive_scan.atr_baseline** – ATR level corresponding to a 1× factor.
* **adaptive_scan.max_factor** – cap multiplier for batch size and scan rate.
* **symbol_refresh_minutes** – minutes before the symbol queue is refreshed.
* **symbol_filter** - filters by minimum volume, 24h change percentile, spread and correlation.
* **skip_symbol_filters** – bypass the volume and spread checks and use the provided symbol list as-is.
* **symbol_score_weights** – weights for volume, spread, change, age and liquidity. The weights must sum to a positive value.

```yaml
symbol_score_weights:
  volume: 0.25
  spread: 0.1
  change: 0.45
  liquidity: 0.15
  latency: 0.03
  age: 0.02
```
* **uncached_volume_multiplier** – extra volume factor applied when a pair is missing from `cache/liquid_pairs.json`.
* **min_symbol_age_days** – skip newly listed pairs.
* **min_symbol_score** – minimum score required for trading.
* **top_n_symbols** – maximum number of active markets.
* **max_age_days**, **max_change_pct**, **max_spread_pct**, **max_latency_ms**, **max_vol** – additional scanning limits.
* **use_numba_scoring** – enable numba acceleration for symbol scoring when available.
* **arbitrage_enabled** – compare CEX and Solana DEX prices each cycle.
* **solana_scanner.gecko_search** – query GeckoTerminal to verify volume for new Solana tokens.

### Risk Parameters
* **risk** – default stop loss, take profit and drawdown limits. `min_volume` is set to `0.1` to filter thin markets. The stop is 1.5× ATR and the take profit is 3× ATR by default.
* **trade_size_pct** – percent of capital used per trade.
* **max_open_trades** – maximum simultaneous open trades.
* **max_slippage_pct** – slippage tolerance for orders.
* **liquidity_check**/**liquidity_depth** – verify order book depth.
* **weight_liquidity** – scoring weight for available pool liquidity on Solana pairs.
* **volatility_filter** - skips trading when ATR is too low or funding exceeds `max_funding_rate`. The minimum ATR percent is `0.0001`.
* **sentiment_filter** - checks the Fear & Greed index and Twitter sentiment to avoid bearish markets.
* **sl_pct**/**tp_pct** – defaults for Solana scalper strategies.
* **mempool_monitor** – pause or reprice when Solana fees spike.
* **gas_threshold_gwei** – abort scalper trades when priority fees exceed this.
* **min_cooldown** – minimum minutes between trades.
* **cycle_bias** – optional on-chain metrics to bias trades.
* **min_expected_value** – minimum expected value for a strategy based on
  historical stats.
* **default_expected_value** – fallback EV when no stats exist. When unset,
  the expected value check is skipped.
* **drawdown_penalty_coef** – weight applied to historical drawdown when
  scoring strategies.
* **safety** – kill switch thresholds and API error limits.
* **scoring** – windows and weights used to rank strategies.
* **exec** – advanced order execution settings.
* **exits** – default take profit and stop loss options.

### Strategy and Signals
* **strategy_allocation** – capital split across strategies.
* **strategy_evaluation_mode** – how the router chooses a strategy.
* **ensemble_min_conf** – minimum confidence required for a strategy to
  participate in ensemble evaluation.
* **voting_strategies**/**min_agreeing_votes** – strategies used for the voting router.
* **exit_strategy** – partial profit taking and trailing stop logic. The trailing stop follows price by 2% after at least 1% gain.
* **micro_scalp** – EMA settings plus volume z-score and ATR filters for the scalp bot.
  Supports tick-level aggregation, optional mempool fee checks and order-book
  imbalance filtering with an optional penalty. Set `trend_filter` or
  `imbalance_filter` to `false` to bypass the trend or order book checks.
* **pattern_timeframe** – optional candle interval used by the bounce scalper to
  confirm engulfing or hammer patterns. Defaults to `1m`.
* **trigger_once** – bypass the cooldown and win-rate filter for one bounce scalper cycle.
* **cooldown_enabled** – disable to ignore the cooldown and win-rate check.
* **breakout** – Bollinger/Keltner squeeze with `donchian_window`,
  `vol_confirmation`/`vol_multiplier`, `setup_window`, `trigger_window` and a
  `risk` section for stop sizing.
* **grid_bot.volume_filter** – require a volume spike before entering a grid
  trade. Turning this off increases trade frequency.
* **grid_bot.dynamic_grid** – realign grid steps when the 1h ATR% changes by
  more than 20%. Spacing is derived from ``spacing_pct = max(0.3, 1.2 × ATR%)``
  and is enabled by default.
* **grid_bot.num_levels** – number of grid levels (default ``6``).
* **grid_bot.cooldown_bars** – bars between fills (default ``2``).
* **grid_bot.max_active_legs** – maximum simultaneously open grid legs (default ``8``).
* **grid_bot.spacing_factor** – base spacing multiplier (default ``0.3``).
* **grid_bot.min_range_pct** – required price range percentage (default ``0.0005``).
```yaml
grid_bot:
  dynamic_grid: true
  atr_period: 14
  volume_filter: true
```

* **sniper_bot.atr_window**/**sniper_bot.volume_window** – windows for ATR and
  volume averages when detecting news-like events.

* **atr_normalization** – adjust signal scores using ATR.
```python
score, direction, atr = breakout_bot.generate_signal(lower_df, cfg, higher_df)
size = risk_manager.position_size(score, balance, lower_df, atr=atr)
```
* **ml_signal_model**/**signal_weight_optimizer** – blend strategy scores with machine-learning predictions.
* **signal_threshold**, **min_confidence_score**, **min_consistent_agreement** – thresholds for entering a trade. `min_confidence_score` and `signal_fusion.min_confidence` default to `0.05`.
* **regime_timeframes**/**regime_return_period** – windows used for regime detection.
* **regime_overrides** – optional settings that replace values in the `risk` or strategy sections when a specific regime is active.
```yaml
regime_overrides:
  trending:
    risk:
      sl_mult: 1.2
      tp_mult: 2.5
```
* **twap_enabled**, **twap_slices**, **twap_interval_seconds** – settings for time-weighted order execution.
* **optimization** – periodic parameter optimisation.
* **portfolio_rotation** – rotate holdings based on scoring metrics.
* **arbitrage_enabled** – enable cross-exchange arbitrage features.
* **scoring_weights** - weighting factors for regime confidence, symbol score and volume metrics.
* **signal_fusion** - combine scores from multiple strategies via a `fusion_method`.
* **strategy_router** - maps market regimes to lists of strategy names. Each regime also accepts a `<regime>_timeframe` key (e.g. `trending_timeframe: 1h`, `volatile_timeframe: 1m`).
* **mode_threshold**/**mode_degrade_window** - degrade to manual mode when auto selection underperforms.
* **meta_selector**/**rl_selector** – experimental strategy routers.
* **bandit_router** – Thompson sampling router that favors historically profitable strategies.
* **bandit** – Thompson sampling selector; tune `explore_pct` for exploration.
* **mode** – `auto` or `manual` evaluation of strategies.
* **parallel_strategy_workers** – strategies evaluated concurrently when
  ranking candidates.
* **second_place_csv** – file that records the runner‑up from each
  parallel evaluation cycle.
* **ensemble_min_conf** – minimum score required for a strategy to be
  ranked in ensemble mode.

To enable the Thompson sampling router add the following to `crypto_bot/config.yaml`:

```yaml
bandit_router:
  enabled: true
```

When `strategy_evaluation_mode` is set to `ensemble`, strategies mapped
to the current regime are scored concurrently. The helper `run_candidates`
ranks them by `score × edge` and executes the best result. Details about
the second‑highest strategy are written to the CSV file defined by
`second_place_csv`.
#### Bounce Scalper
The bounce scalper looks for short-term reversals when a volume spike confirms multiple down or up candles. Scores are normalized with ATR and trades use ATR-based stop loss and take profit distances. Each signal observes `min_cooldown` before re-entry. Set `pattern_timeframe` (default `1m`) to fetch a separate candle interval for confirming engulfing or hammer patterns. When in cooldown the scalper only signals if the recent win rate falls below 50%, effectively skipping the cooldown during a drawdown. Set `cooldown_enabled` to `false` to disable this behaviour.

```yaml
bounce_scalper:
  pattern_timeframe: 5m  # confirm patterns using 5-minute candles
  cooldown_enabled: false  # disable cooldown checks
min_cooldown: 2          # minutes between entries
```

Calling `bounce_scalper.trigger_once()` bypasses the filter for a single cycle.

#### Mean Bot
The mean reversion bot now incorporates an ADX trend filter to avoid
counter‑trend trades. Its RSI thresholds are scaled according to recent
volatility, and you can optionally blend the final score with a machine
learning prediction. Enable the weighting in `crypto_bot/config.yaml`:

```yaml
mean_bot:
  ml_enabled: true
```
The bot only opens positions when the current 20-bar Bollinger bandwidth is
below its 20-bar median, reducing trades during ranging periods and improving
the win rate.

### Data and Logging
* **timeframe** – base interval for most indicators (default `15m`).
* **timeframes** – list of additional intervals cached for reuse by strategies.
* **scalp_timeframe** – short interval used by the scalping bots.
* **ohlcv_snapshot_frequency_minutes**/**ohlcv_snapshot_limit** – OHLCV caching options. A separate cache is maintained for each timeframe listed in `timeframes`.
* **timeframe**, **timeframes**, **scalp_timeframe** – candle intervals used for analysis. Default `timeframe` is `15m` and `timeframes` include `1m`, `5m`, `15m`, `1h`, and `4h`.
* **ohlcv_snapshot_frequency_minutes**/**ohlcv_snapshot_limit** – OHLCV caching options. The snapshot limit defaults to `500`.
* **loop_interval_minutes** – delay between trading cycles.
* **ohlcv_timeout**, **max_concurrent_ohlcv**, **max_ohlcv_failures** – limits for candle requests.
* **max_parallel** – number of markets processed concurrently.
* **log_to_google** – export trades to Google Sheets.
* **telegram** – bot token, chat ID and trade notifications. Optional
  **status_updates** and **balance_updates** flags control startup and
  balance alerts.
* **balance_change_threshold** – delta for Telegram balance alerts.
* **balance_poll_mod** – how often to poll balance between trades.
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

The built-in Telegram interface is provided by the `TelegramBotUI` class in
`crypto_bot.telegram_bot_ui`.

1. Open `crypto_bot/config.yaml` and fill in the `telegram` section:

   ```yaml
   telegram:
     token: your_telegram_token
     chat_id: your_chat_id
     chat_admins: your_chat_id
     trade_updates: true
   ```

   The bot reads the chat ID and token from `config.yaml` (not
   `user_config.yaml`). Set `trade_updates` to `false` to disable trade entry
   and exit messages.
     trade_updates: true  # set false to disable trade notifications
   ```

   The bot reads these values only from `config.yaml`. Disable
   `trade_updates` if you don't want trade entry and exit messages.
   Set `chat_admins` to a comma-separated list of Telegram chat IDs allowed to
   control the bot. You can also provide this list via the `TELE_CHAT_ADMINS`
   environment variable.
2. Send `/start` to your bot so it can message you. Use `/menu` at any time to
   open an interactive button menu—**Start**, **Stop**, **Status**, **Log**,
   **Rotate Now**, **Toggle Mode**, **PnL**, **Trades**, **Edit Config**,
   **Signals** and **Balance**—for quick interaction.
3. You can also issue these commands directly:
   - `/signals` – show the latest scored assets
   - `/balance` – display your current exchange holdings
   - `/trades` – summarize executed trades
   - `/panic_sell` – exit all open positions immediately (paper or live)
   - HTTP `POST /close-all` – trigger the same exit via the web server,
     also works in paper trading mode.
4. If you see `Failed to send message: Not Found` in the logs, the chat ID or
   token is likely incorrect or the bot lacks permission to message the chat.
   Double-check the values in `config.yaml` and ensure you've started a
  conversation with your bot.

#### Troubleshooting

Before running the bot, run `python tools/test_telegram.py` to send a
test message using the credentials in `crypto_bot/config.yaml` or the
`TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` environment variables. The script
calls `crypto_bot.utils.telegram.send_test_message()` under the hood.
If the call fails or you do not receive a message, check for these common issues:

* **Invalid config values** – `telegram.token` or `telegram.chat_id` still
  contain placeholders in `crypto_bot/config.yaml`.

* **Incorrect token** – the API token was mistyped or revoked.
* **Wrong chat ID** – the bot does not have permission to message that chat.
* **Bot not started** – you have not sent `/start` to your bot yet.
* **Network restrictions** – firewalls or proxies are blocking Telegram.

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
weight_liquidity: 0.0        # symbol score weight for pool liquidity
twap_enabled: false          # split large orders into slices
twap_slices: 4               # number of slices when TWAP is enabled
twap_interval_seconds: 10    # delay between TWAP slices
timeframe: 15m               # candles for regime detection
scalp_timeframe: 1m          # candles for micro_scalp/bounce_scalper
loop_interval_minutes: 0.5   # wait time between trading cycles
force_websocket_history: false  # set true to disable REST fallback
max_ws_limit: 50             # skip WebSocket when request exceeds this
ohlcv_timeout: 300            # request timeout for OHLCV fetches
max_concurrent_ohlcv: 4      # limit simultaneous OHLCV fetches
force_websocket_history: true  # set false to enable REST fallback
max_ws_limit: 200            # skip WebSocket when request exceeds this
ohlcv_timeout: 120            # request timeout for OHLCV fetches
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
The updater automatically determines how many candles are missing from the
cache, so even when `limit` is large it only requests the data required to fill
the gap, avoiding needless delays.
The bot caches the last candle timestamp for open positions and skips updating
their history until a new bar appears.
The `metrics` section enables recording of cycle summaries to the specified CSV
file for later analysis.
`scalp_timeframe` sets the candle interval specifically for the micro_scalp
and bounce_scalper strategies while `timeframe` covers all other analysis.

When `use_websocket` is enabled the bot relies on `ccxt.pro` for realtime
streaming data. Install it alongside the other requirements or disable
websockets if you do not have access to `ccxt.pro`.
When OHLCV streaming returns fewer candles than requested the bot calculates
how many bars are missing and fetches only that remainder via REST. This
adaptive limit keeps history current without waiting for a full response.
Enable this fallback by setting `force_websocket_history` to `false`.
Large history requests skip streaming entirely when `limit` exceeds
`max_ws_limit`.
Increase this threshold in `crypto_bot/config.yaml` when large history
requests should still use WebSocket. For example set
`max_ws_limit: 200` if you regularly request 200 candles.

During the startup scan the bot always loads historical candles over REST
regardless of the WebSocket setting. It calls `fetch_ohlcv` for up to
`scan_lookback_limit` candles per pair (700 by default on Kraken) to build the
cache before realtime updates begin over WebSocket.

The client now records heartbeat events and exposes `is_alive(conn_type)` to
check if a connection has received a heartbeat within the last 10 seconds. Call
`ping()` periodically to keep the session active.

Example usage for Kraken WebSockets:

```python
from crypto_bot.execution.kraken_ws import KrakenWSClient

ws = KrakenWSClient()
ws.subscribe_orders(["BTC/USD"])  # open_orders channel
ws.subscribe_book("BTC/USD", depth=10, snapshot=True)
ws.add_order("BTC/USD", "buy", qty)
ws.cancel_order("ORDERID")
ws.add_order("BTC/USD", "buy", 0.01)
ws.cancel_order("OABCDEF", ["BTC/USD"])
ws.subscribe_instruments()  # stream asset and pair details
ws.close()  # gracefully close the websockets when done
```

To stream ticker data use `subscribe_ticker`. The optional `event_trigger`
parameter controls which events push updates and defaults to `"trades"`. The
`snapshot` flag requests an initial snapshot and defaults to `True`.

```python
# Request ticker updates triggered by book changes without an initial snapshot
ws.subscribe_ticker(["ETH/USD"], event_trigger="book", snapshot=False)
```
To stream candlestick data use `subscribe_ohlc`. The helper
`parse_ohlc_message` converts the raw payload into `[timestamp, open, high,
low, close, volume]` where `timestamp` is the `interval_begin` field converted
to a Unix epoch in milliseconds.

```python
ws.subscribe_ohlc("ETH/USD", interval=1)

msg = ...  # read from ws.public_ws
candle = parse_ohlc_message(msg)
if candle:
    ts, o, h, l, c, volume = candle
    print(ts, o, h, l, c, volume)
```
Call `unsubscribe_ohlc("ETH/USD", interval=1)` to stop receiving updates.

`subscribe_book` streams the order book for the given pair. `depth` sets how many levels are sent, while `snapshot` requests an initial book snapshot before updates.

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
client.close()
```

#### Trade Updates

Subscribe to the public trades channel to monitor real-time fills.
Refer to Kraken's v2 WebSocket documentation for message details:
<https://docs.kraken.com/websockets-v2/#tag/Trading>.

```python
ws.subscribe_trades(["BTC/USD"], snapshot=False)

msg = ...  # read from ws.public_ws
trade = parse_trade_message(msg)
if trade:
    ts, side, price, volume = trade
    print(ts, side, price, volume)

ws.unsubscribe_trades(["BTC/USD"])
```

#### Level 3 Order Updates

Subscribe to the full depth feed using `subscribe_level3`. The call requires a
session token obtained from Kraken's `GetWebSocketsToken` REST endpoint. Depth
values of `10`, `100` or `1000` are supported.

```python
import json
from crypto_bot.execution.kraken_ws import KrakenWSClient

ws = KrakenWSClient(ws_token="your_ws_token")
ws.subscribe_level3("BTC/USD", depth=100)

def handle(msg: str):
    data = json.loads(msg)
    if data.get("channel") == "level3":
        book = data["data"][0]
        for order in book.get("bids", []):
            # each order dict contains event, order_id, limit_price, order_qty
            print(order)
```

Binance.US is not recommended because of API limitations.

### Automatic Market Scanning

When `scan_markets` is set to `true` and the `symbols` list is empty, the bot
loads all active Kraken trading pairs at startup. Pairs listed under
`excluded_symbols` are skipped. Disable this behaviour by setting
`scan_markets` to `false`. When `scan_in_background` is `true` the scan runs as
a background task so trading phases start immediately. Set it to `false` to
wait for scanning to complete before trading begins.

```yaml
scan_markets: true
scan_in_background: true
symbols: []            # automatically populated
solana_symbols: []     # base tokens traded on Solana
excluded_symbols: [ETH/USD]
exchange_market_types: ["spot"]  # options: spot, margin, futures
min_symbol_age_days: 2           # skip pairs with less history
symbol_batch_size: 50            # symbols processed per cycle
scan_lookback_limit: 700         # candles loaded during startup
cycle_lookback_limit: 150        # candles fetched each cycle
max_spread_pct: 4.0              # skip pairs with wide spreads
```

To avoid loading every market on startup, populate `symbols` with the
top 200 pairs by volume from `tasks/refresh_pairs.py`. Only set
`scan_markets: true` when you need to evaluate the entire exchange.

`exchange_market_types` filters the discovered pairs by market class. The bot
also skips newly listed pairs using `min_symbol_age_days`.
Symbols are queued by score using a priority deque and processed in
batches controlled by `symbol_batch_size`. When the queue drops below this
size it is automatically refilled with the highest scoring symbols.
Candidates are stored in a priority queue sorted by their score so the highest
quality markets are scanned first. Each cycle pulls the next `symbol_batch_size`
symbols from this queue and refills it when empty.

When `adaptive_scan.enabled` is true the bot calculates the average ATR of the
filtered markets. The batch size and delay between cycles are multiplied by
`avg_atr / atr_baseline` up to `max_factor`. This increases scanning frequency
during volatile periods.

```yaml
adaptive_scan:
  enabled: true
  atr_baseline: 0.01
  max_factor: 5.0
```

OHLCV data for these symbols is now fetched concurrently using
`load_ohlcv_parallel`, greatly reducing the time needed to evaluate
large symbol lists.

Each candidate pair is also assigned a score based on volume, recent price
change, bid/ask spread, age and API latency. The weights and limits for this
calculation can be tuned via `symbol_score_weights`, `max_vol`,
`max_change_pct`, `max_spread_pct`, `max_age_days` and `max_latency_ms` in
`config.yaml`. All scoring weights must sum to a positive value. Only symbols with a score above `min_symbol_score` are included
in trading rotations.
## Symbol Filtering

The bot evaluates each candidate pair using Kraken ticker data. By
setting options under `symbol_filter` you can weed out illiquid or
undesirable markets before strategies run. Set `skip_symbol_filters: true`
to use the provided list without any filtering:

```yaml
symbol_filter:
  min_volume_usd: 100
  volume_percentile: 10          # keep pairs above this volume percentile
  change_pct_percentile: 5       # require 24h change in the top movers
  max_spread_pct: 4              # allow spreads up to 4%
  uncached_volume_multiplier: 1.5  # extra volume when not cached
  correlation_window: 30         # days of history for correlation
  max_correlation: 0.9           # drop pairs above this threshold
  correlation_max_pairs: 100     # limit pairwise correlation checks
  kraken_batch_size: 100         # max symbols per fetch_tickers call
  http_timeout: 10               # seconds for fallback /Ticker requests
  ticker_retry_attempts: 3       # number of fetch_tickers retries
  log_ticker_exceptions: false   # include stack traces when true
```
`setup_window` controls how many candles of ticker history are gathered before
a symbol is eligible to trade, while `trigger_window` defines the period after
a setup is detected during which entry conditions must appear.  Each strategy
can also include a `risk` section such as `max_concurrent` or
`daily_loss_cap` to cap simultaneous positions and total daily losses.
WebSocket streaming is enabled by default when scanning. Set
`use_websocket: false` to force REST polling instead. You can disable
WebSocket just for ticker scanning by adding
`exchange.options.ws_scan: false` to your configuration while leaving
`use_websocket: true` for trading. When using the REST fallback the bot
requests tickers in batches controlled by
`symbol_filter.kraken_batch_size` to avoid Kraken's occasional `520`
errors. The public `/Ticker` calls also obey
`symbol_filter.http_timeout`.
Pairs passing these checks are then scored with `analyze_symbol` which
computes a strategy confidence score. Only the highest scoring symbols
are traded each cycle.

### Liquid Pairs Worker

The `tasks/refresh_pairs.py` script fetches the most liquid markets from the
configured exchange using `ccxt` and stores them in `cache/liquid_pairs.json`.
The file now contains a mapping of symbol to the timestamp when it last passed
the liquidity screen. This cache lets the trading bot skip illiquid pairs during
market scans.
By default the worker refreshes the file every **6 hours**. Change the interval
under `pairs_worker.refresh_interval` in `crypto_bot/config.yaml` and restart the
worker to apply the new schedule.
You can also limit the markets saved in the cache by defining
`allowed_quote_currencies` and `blacklist_assets` under `refresh_pairs`.
Leaving `allowed_quote_currencies` empty allows any trading pair:

```yaml
refresh_pairs:
  min_quote_volume_usd: 1000000
  refresh_interval: 6h
  top_k: 40
  secondary_exchange: coinbase
  allowed_quote_currencies: []
  blacklist_assets: []
```
Run it manually whenever needed:

```bash
python tasks/refresh_pairs.py --once
```
Removing the `--once` flag keeps it running on the configured interval.
To automate updates you can run the script periodically via cron:

```cron
0 * * * * cd /path/to/coinTrader2.0 && /usr/bin/python3 tasks/refresh_pairs.py
```
Delete `cache/liquid_pairs.json` to force a full rebuild on the next run.

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
`crypto_bot/logs/strategy_stats.json` (automatically produced from
`strategy_performance.json`) and the detailed performance records in
`crypto_bot/logs/strategy_performance.json`. When the bot is stopped a form
lets you select the execution mode (dry run or live) before launching.

## Log Files

All runtime information is written under `crypto_bot/logs`. Important files
include:

- `bot.log` – main log file containing startup events, strategy choices and all
  decision messages.
- `trades.csv` – CSV export of every executed order used by the dashboard and
  backtester. Entries may represent long or short positions: a `buy` side opens
  or closes a short while a `sell` side opens or closes a long. Stop orders are
  logged here as well with an `is_stop` flag so they can be filtered out from
  performance calculations. Open positions are reconstructed by scanning the
  rows sequentially and pairing each entry with the next opposite side.
- `strategy_stats.json` – summary statistics of win rate, PnL and other metrics
  generated automatically from `strategy_performance.json`.
- `strategy_performance.json` – list of individual trades grouped by regime and
  strategy with fields like `pnl` and timestamps.
- `metrics.csv` – per cycle summary showing how many pairs were scanned,
  how many signals fired and how many trades executed.
- `weights.json` – persistent optimizer weights saved after each update
  at `crypto_bot/logs/weights.json`.
- `second_place.csv` – the runner‑up strategy from each evaluation cycle.

Example short trade:

```csv
symbol,side,amount,price,timestamp
XBT/USDT,sell,0.1,25000,2024-05-01T00:00:00Z
XBT/USDT,buy,0.1,24000,2024-05-02T00:00:00Z
```

This opens a short at 25,000 and covers at 24,000 for a profit of
`(25000 - 24000) * 0.1 = 100` USDT.

### Statistics File Structure

`strategy_performance.json` stores raw trade records nested by market regime and
strategy. Example snippet:

```json
{
  "trending": {
    "trend_bot": [
      {
        "symbol": "XBT/USDT",
        "pnl": 1.2,
        "entry_time": "2024-01-01T00:00:00Z",
        "exit_time": "2024-01-01T02:00:00Z"
      }
    ]
  }
}
```

`strategy_stats.json` contains aggregated statistics per strategy such as win
rate and average PnL. It is produced automatically from
`strategy_performance.json`:

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
2024-02-12 15:04:02 - INFO - Strategy router selected grid_bot for XBT/USDT
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
If `gas_threshold_gwei` is set, the scalper aborts the swap entirely when
the priority fee exceeds this limit.

## Solana Meme-Wave Sniper

This module watches for new liquidity pools on Solana and attempts to buy
into meme tokens before the crowd. Events from a Helius endpoint are
filtered through safety checks, scored, and executed using Jupiter quotes
bundled via Jito. A Twitter sentiment score can boost the ranking when the
tweet volume is high.

### Configuration

Add a `meme_wave_sniper` section to `crypto_bot/config.yaml`:

Set `HELIUS_KEY` in `crypto_bot/.env` or as an environment variable. The pool
URL should reference this key so Helius can authorize the requests:

```yaml
meme_wave_sniper:
  enabled: true
  pool:
    url: https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_KEY}
    interval: 5
    websocket_url: wss://atlas-mainnet.helius-rpc.com/?api-key=${HELIUS_KEY}
    raydium_program_id: EhhTK0i58FmSPrbr30Y8wVDDDeWGPAHDq6vNru6wUATk
  scoring:
    weight_liquidity: 1.0
    weight_tx: 1.0
    weight_social: 0.5
  safety:
    min_liquidity: 10
  risk:
    max_concurrent: 20
    daily_loss_cap: 1.5
  execution:
    dry_run: true

```
Set the `HELIUS_KEY` environment variable with your Helius API key.

### Flow

```text
PoolWatcher -> Safety -> Score -> RiskTracker -> Executor -> Exit
```

Sniping begins immediately at startup. The initial symbol scan now runs in the
background so new pools can be acted on without waiting for caches to fill.

API requirements: [Helius](https://www.helius.xyz/) for pool data,
[Jupiter](https://jup.ag/) for quotes, [Jito](https://www.jito.network/) for
bundle submission, and a [Twitter](https://developer.twitter.com/) token for
sentiment scores.

### Monitoring Raydium Pools via WebSockets

Raydium also streams pool creation events over WebSockets. To watch these in
real time:

1. Obtain a Helius API key from your dashboard.
2. Set `meme_wave_sniper.pool.websocket_url` in `crypto_bot/config.yaml` to
   `wss://mainnet.helius-rpc.com/?api-key=YOUR_KEY`.
   `atlas-mainnet.helius-rpc.com` is only available for Business/Professional
   plans; standard tiers should use the mainnet URL shown above.
3. Run `python -m crypto_bot.solana.pool_ws_monitor`.

`pool_ws_monitor.py` subscribes to the Raydium program and prints each update:

```python
import asyncio
from crypto_bot.solana.pool_ws_monitor import watch_pools

async def main():
    async for event in watch_pools():
        print(event)

asyncio.run(main())
```

### Backtesting

The `BacktestRunner` class in `crypto_bot.backtest.backtest_runner` can evaluate
different stop‑loss and take‑profit percentages and reports the PnL,
maximum drawdown and Sharpe ratio for each combination.

```python
from crypto_bot.backtest.backtest_runner import BacktestRunner

runner = BacktestRunner('XBT/USDT', '15m', since=0)
results = runner.run_grid(
    stop_loss_range=[0.01, 0.02],
    take_profit_range=[0.02, 0.04],
)
print(results.head())  # best combo appears first
```
For Solana pairs ending with `/USDC`, backtesting automatically pulls up to
`1000` candles from GeckoTerminal so longer histories are available.
The resulting statistics are written automatically to
`crypto_bot/logs/strategy_stats.json`. The home page indicates whether the bot
is running so you can quickly see if it has stopped.

## PhaseRunner

`PhaseRunner` orchestrates the main trading cycle by executing a list of async
phases in sequence. Each phase receives a shared `BotContext` object so they can
exchange data as the run progresses. The loop inside `crypto_bot.main` relies on
this runner to fetch candidates, update caches, analyse opportunities, execute
orders and manage exits on every iteration.

```python
from crypto_bot.phase_runner import PhaseRunner, BotContext

async def fetch(ctx):
    ...  # gather symbols or data

async def analyse(ctx):
    ...  # compute signals

async def trade(ctx):
    ...  # place orders

runner = PhaseRunner([fetch, analyse, trade])
ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={})
await runner.run(ctx)
```

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

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite to verify your environment:

```bash
pytest -q
```

## Testing

The repository includes an automated test suite. Some tests rely on optional
packages such as `numpy`, `pandas`, and `ccxt`.  Lightweight stubs allow the suite to run
in very small environments, but the **full** set of tests requires the
dependencies listed in `requirements.txt`.

Set up the environment by running the provided script:

```bash
bash codex/setup.sh  # installs system and Python dependencies
```

Alternatively you can install the Python packages manually:

```bash
pip install -r requirements.txt
```

If `pytest` fails with a `ModuleNotFoundError`, ensure the packages from the
requirements file are installed.  After the dependencies are available, execute

```bash
pytest -q
```

## Troubleshooting

High `max_concurrent_ohlcv` values combined with short `ohlcv_timeout`
settings can overload the exchange and lead to failed candle fetches.
Increase `ohlcv_timeout` to give each request more time and lower
`max_concurrent_ohlcv` if errors continue.

This project is provided for educational purposes only. Use it at your own risk, and remember that nothing here constitutes financial advice.


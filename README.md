# Crypto Trading Bot

This project provides a modular hybrid cryptocurrency trading bot capable of operating on centralized exchanges like Coinbase Advanced Trade or Kraken and on Solana DEXes via the Jupiter aggregator.

## Features

* Regime detection using EMA, ADX, RSI and Bollinger Band width
* Strategy router that picks the best approach for trending, sideways, breakout or volatile markets
* Fast-path dispatcher that jumps straight to the breakout or trend_bot on strong signals
* trend_bot and grid_bot for CEXs plus sniper_bot and dex_scalper strategies on Solana
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
* Optional machine learning features download pretrained models from Supabase; install `cointrader-trainer` only if you plan to train new models
* Trade execution helpers poll for order status on CEXs or transaction confirmation on Solana with a 60s timeout. Partial fills are handled automatically via the optional `poll_timeout` parameter.

On-chain DEX execution submits real transactions when not running in dry-run mode.

## Wallet

`Wallet` acts as a simple balance and PnL tracker in dry-run mode. It keeps a
cash balance along with realized and unrealized profit, letting you test trading
logic without contacting a live exchange.

```python
from crypto_bot.paper_wallet import PaperWallet as Wallet
import time

wallet = Wallet(1000.0)
wallet.open("BTC/USDT", "buy", 1.0, 100.0)  # buy
wallet.close("BTC/USDT", 1.0, 110.0)        # sell

while True:
    current = 110.0  # replace with a real price feed
    total = wallet.balance + wallet.realized_pnl + wallet.unrealized(current)
    print(f"Total balance: ${total:.2f}")
    time.sleep(60)
```

The snippet above initializes a dry-run wallet, demonstrates buy and sell calls,
and periodically prints the combined total balance.

## Regime Classifier

The bot selects a strategy by first classifying the current market regime. The
`classify_regime` function computes EMA, ADX, RSI and Bollinger Band width to
label conditions as `trending`, `sideways`, `breakout`, `mean-reverting` or
`volatile`. At least **200** candles are recommended for these indicators to
be calculated reliably. When fewer than **5** candles are available the
`volatile`. While **200** candles are recommended for these indicators to
be calculated reliably, the classifier can operate with less history.
`initial_history_candles` or `scan_lookback_limit` controls how many bars are
fetched during startup and may be set below 200 if quicker initialization is
preferred. When fewer than **5** candles are available the
system assumes a `breakout` regime to avoid missing early momentum.
With 5 to 199 candles the classifier still runs but accuracy may be
reduced. Strategies may
operate on different candle intervals, so the loader keeps a multi‑timeframe
cache populated for each pair. The `timeframes` list in
`crypto_bot/config.yaml` defines which intervals are stored and reused across
the various bots.

### OHLCV bootstrap and defer

Specify higher timeframes in the `ohlcv` section of the config to control how
the candle cache warms up. `bootstrap_timeframes` (default ['1h']) are loaded
on startup, and the bot begins scoring once these frames are ready. Additional
`defer_timeframes` such as ['4h', '1d'] warm in the background so
initialization is not blocked. The legacy `trading.backfill.warmup_high_tf`
option has been removed.

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
By default the ML model runs once at least **5** candles are available (`ml_min_bars` in `crypto_bot/regime/regime_config.yaml`).
You can replace that module with your own encoded model if desired.
When enough history is present the ML probabilities are blended with the
rule-based result using `ml_blend_weight` from `regime_config.yaml`.

The regime configuration exposes additional tuning parameters:

* **adx_trending_min** – ADX threshold for the trending regime. Defaults to `10`.
* **rsi_mean_rev_min** / **rsi_mean_rev_max** – RSI range for mean-reversion. Defaults to `30`‑`70`.
* **ml_blend_weight** – blend ratio for combining ML and indicator scores. Defaults to `0.8`.
* **breakout_volume_mult** – volume multiplier for breakout detection.
* **score_weights** – weighting factors for regime probabilities when patterns
  are detected.
* **pattern_min_conf** – minimum pattern confidence required to apply a score
  weight.
* **bull_fng** – Fear & Greed index level considered bullish.
* **atr_baseline** – ATR level corresponding to a 1× score factor.

When the configured `timeframe` or `scalp_timeframe` is **sub‑minute**, the router
switches to high‑frequency values prefixed with `hft_`:

```yaml
hft_adx_trending_min: 10
hft_rsi_mean_rev_min: 30
hft_rsi_mean_rev_max: 70
hft_ml_blend_weight: 0.8
```

Example:

```yaml
timeframe: 30s
scalp_timeframe: 15s
```

With the above settings, the HFT defaults are applied automatically. Override
any of them via environment variables:

```bash
export HFT_ADX_TRENDING_MIN=12
export HFT_RSI_MEAN_REV_MIN=25
export HFT_RSI_MEAN_REV_MAX=75
export HFT_ML_BLEND_WEIGHT=0.9
```

#### Model Loader Environment

Trained LightGBM models may be stored in Supabase using a stable layout:

```
bucket: models
prefix: models/regime/{SYMBOL}/…
```

The loader searches this path for a `LATEST.json` manifest to resolve the
current model. Models are fetched automatically from Supabase when
`LATEST.json` exists. The `cointrader-trainer` package is only required if you
intend to train and upload new models; otherwise, the embedded base64 model
bundled with the bot is used as a fallback.

Configure the loader with the following environment variables:

```ini
SUPABASE_URL=
# One of the following keys must be set to download ML regime models
SUPABASE_SERVICE_ROLE_KEY=
# SUPABASE_KEY=
# SUPABASE_API_KEY=
CT_MODELS_BUCKET=models
CT_REGIME_PREFIX=models/regime
CT_SYMBOL=BTCUSDT
```

## Fast-Path Checks

The router performs quick checks for exceptionally strong setups before running
the full regime classifier. When the Bollinger Band width over a 20 candle
window drops below **0.05** and volume is more than **5×** the average,
`breakout_bot` is called immediately. If the ADX from the same window exceeds
**35**, the router dispatches straight to `trend_bot`. When a breakdown pattern
appears and the most recent volume is well above average, the router jumps
directly to the sniper strategy. These defaults live under
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
   The reinforcement learning strategy selector requires additional packages:
   `gymnasium` and `stable-baselines3`. These are included in `requirements.txt`.
   Install them with `pip install -r requirements.txt` when `rl_selector.enabled`
   is set to `true` in `crypto_bot/config.yaml`. Set `rl_selector.enabled: false`
   if you prefer not to install these extra dependencies.
    Machine learning integrations download pretrained models from Supabase.
    The `cointrader-trainer` package is only needed when training new models;
    runtime usage pulls models automatically as long as `SUPABASE_URL` and
    `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_KEY`/`SUPABASE_API_KEY`) are configured.
2. (Optional) Run `python crypto_bot/wallet_manager.py` to create `user_config.yaml` and enter your API credentials, including your Helius API key.
   `python -m crypto_bot.main` will launch this setup wizard automatically when credentials or `user_config.yaml` are missing.
2. Run `python crypto_bot/wallet_manager.py` to create `user_config.yaml` and enter your API credentials, including your Helius and Supabase API keys.
3. Adjust `crypto_bot/config.yaml` to select the exchange and execution mode.
4. Start the trading bot:
   ```bash
   python -m crypto_bot.main
   ```
   When dry-run mode is selected you will be prompted for the starting USDT balance.
   The console now refreshes with your wallet balance and any active
   trades in real time. Profitable positions are shown in green while
   losing ones appear in red. The monitor prints a single status line
   followed by additional lines for each open trade formatted as
   `Symbol -- entry -- unrealized PnL`.
   The program prints "Bot running..." before the [Monitor] lines.
   Before trading begins the bot performs a full market scan to populate
   its caches. Progress is logged and, when `telegram.bootstrap_updates`
   is enabled, sent to your Telegram chat.
   In dry-run mode order prices are pulled from the latest ticker. If the
   exchange fails to return a price the entry value from the trade log is
   used so profit calculations remain accurate.
   Type `start`, `stop`, `panic sell`, `reload` or `quit` in the terminal to
   control the bot.
   Or launch the web dashboard with:
   ```bash
   python -m frontend.app
   ```
5. Run the meme-wave sniper separately with Raydium v3 integration.
   Profits are automatically converted to BTC. Set `SOLANA_PRIVATE_KEY` and the
   required `HELIUS_API_KEY` or provide a custom `SOLANA_RPC_URL` before launching:
   ```bash
   python -m crypto_bot.solana.runner
   ```

6. Edit `crypto_bot/config.yaml` and reload the settings without restarting the
   bot:

   ```yaml
   risk:
     trade_size_pct: 1.5  # new value
     win_rate_threshold: 0.8
     win_rate_boost_factor: 2.0
   ```

   Save the file and type `reload` in the console or send `/reload` via Telegram
   to apply the changes immediately.

## Examples

`examples/wallet_demo.py` demonstrates the paper trading wallet. It opens a
trade, closes it and prints the final balance and realized PnL.

Run it with:

```bash
python examples/wallet_demo.py
```

Run `python crypto_bot/wallet_manager.py` if you want to create `user_config.yaml` and enter your exchange credentials and Helius API key manually. This step is optional—`python -m crypto_bot.main` launches the same wizard automatically whenever credentials or `user_config.yaml` are missing. Values from `.env` override those stored in `user_config.yaml`. Setting `SECRETS_PROVIDER` (`aws` or `vault`) with `SECRETS_PATH` loads credentials automatically. Provide a `FERNET_KEY` to encrypt sensitive values before they are written to `user_config.yaml`. Without this key the wallet manager stores API secrets unencrypted.
Run `wallet_manager.py` to create `user_config.yaml` and enter your exchange credentials and Helius and Supabase API keys. Values from `.env` override those stored in `user_config.yaml`. Setting `SECRETS_PROVIDER` (`aws` or `vault`) with `SECRETS_PATH` loads credentials automatically. Provide a `FERNET_KEY` to encrypt sensitive values before they are written to `user_config.yaml`. Without this key the wallet manager stores API secrets unencrypted.

## Configuration Files

The bot looks in two locations for settings:

1. **`.env`** – store API keys and other environment variables here. These values override entries loaded by `wallet_manager.py` if you choose to run it.
2. **`crypto_bot/config.yaml`** – general runtime options controlling strategy behaviour and notifications.

### Environment Variables

Create `.env` and fill in your secrets. Example:

```env
EXCHANGE=coinbase              # or kraken
API_KEY=your_key
API_SECRET=your_secret
API_PASSPHRASE=your_coinbase_passphrase_if_needed
# generate with:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FERNET_KEY=your_generated_fernet_key
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id          # chat to receive Telegram notifications
WALLET_ADDRESS=your_wallet_address     # default on-chain wallet for trades
TELE_CHAT_ADMINS=123456,789012         # optional comma separated admin IDs
TELE_CHAT_ADMINS=12345,67890          # comma-separated chat IDs
GOOGLE_CRED_JSON=path_to_google_credentials.json
TWITTER_SENTIMENT_URL=https://api.example.com/twitter-sentiment
FUNDING_RATE_URL=https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=
SECRETS_PROVIDER=aws                     # optional
SECRETS_PATH=/path/to/secret
SOLANA_PRIVATE_KEY="[1,2,3,...]"       # required for Solana trades
# defaults to https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_API_KEY}
SOLANA_RPC_URL=https://devnet.solana.com  # optional custom endpoint
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com  # optional
# SOLANA_RPC_URL=https://api.devnet.solana.com      # devnet example
HELIUS_API_KEY=your_helius_api_key      # required for Jupiter/Helius registry
MORALIS_KEY=your_moralis_api_key       # optional, for Solana scanner
BITQUERY_KEY=your_bitquery_api_key     # optional, for Solana scanner
SUPABASE_URL=https://xyzcompany.supabase.co
# One of these keys must be provided for ML regime models
SUPABASE_SERVICE_ROLE_KEY=your_service_key
# SUPABASE_KEY=your_supabase_key
# SUPABASE_API_KEY=your_api_key
LUNARCRUSH_API_KEY=your_lunarcrush_api_key  # optional, LunarCrush social metrics
token_registry.refresh_interval_minutes=720  # optional cache update interval
```

Kraken WebSocket tokens are fetched automatically using your API key and secret. Set `KRAKEN_WS_TOKEN` in `.env` only if you need to override the fetched token.

`FERNET_KEY` holds the encryption key used by wallet_manager.py when saving credentials. Generate it with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and set in `.env`.

Each credential can be supplied in `.env`, or if missing, will be requested interactively by the setup scripts.

`TELE_CHAT_ADMINS` lets the Telegram bot accept commands from multiple
admin chats. Omit it to restrict control to the single `chat_id` in the
configuration file.

`SUPABASE_URL` and either `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_KEY`, or `SUPABASE_API_KEY` are required for downloading models used by `regime_classifier`.

### Solana token registry

Solana tickers are resolved to mint addresses using a hybrid
Jupiter/Helius registry cached in `cache/token_mints.json`. When GeckoTerminal
and Dexscreener searches return no results, the lookup now falls back to
Helius to fetch metadata for the missing symbol.

### Solana Setup

Example RPC URLs:

```env
# Mainnet
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
# Devnet
# SOLANA_RPC_URL=https://api.devnet.solana.com
```

When using [Helius](https://www.helius.xyz/) endpoints, append `?api-key=${HELIUS_API_KEY}` to the URL:

```env
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_API_KEY}
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
# configure up to two exchanges
primary_exchange: coinbase  # coinbase or kraken
secondary_exchange: kraken
solana_rpc: https://api.mainnet-beta.solana.com
execution_mode: dry_run  # or live
mode: auto               # router chooses CEX or on-chain based on token mints
use_websocket: true
kraken:
  use_private_ws: false  # set true only if private WS feeds are required
telegram:
  token: your_telegram_token
  chat_id: your_chat_id
  trade_updates: true
  status_updates: true
  bootstrap_updates: false
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
* **use_websocket** – enable WebSocket data via `KrakenWSClient`.
* **force_websocket_history** – disable REST fallbacks when streaming (default: true).
* **kraken.use_private_ws** – enable private WebSocket feeds for trading; REST-only trading continues when `false`.
* **max_ws_limit** – skip WebSocket OHLCV when `limit` exceeds this value.
* **exchange_market_types** – market types to trade (spot, margin, futures).
* **preferred_chain** – chain used for on-chain swaps (e.g. `solana`).
* **wallet_address** – destination wallet for DEX trades.
* **solana_slippage_bps** – slippage tolerance for on-chain conversions.
* **auto_convert_quote** – token used when converting idle balances (default `USDC`).
* **onchain_min_volume_usd** – minimum 24h volume for on-chain pairs (default `10_000_000`). This value can be overridden in `config.yaml`.
* **symbol**/**symbols** – pairs to trade when not scanning automatically.
* **scan_markets** – load all exchange pairs when `symbols` is empty (enabled by default).
* **scan_in_background** – start the initial scan in the background so trading can begin immediately.
* **excluded_symbols** – markets to skip during scanning.
* **onchain_symbols** – base tokens traded on-chain. Tickers are automatically
  resolved to mint addresses using a hybrid Jupiter/Helius registry cached at
  `cache/token_mints.json`. Each entry is appended with the quote defined by
  `onchain_default_quote` (defaults to `USDC`). If a ticker isn't found in the
  registry, the base value must be a valid mint address. If your configuration
  still contains `solana_symbols`, rename that section to `onchain_symbols`.
  On-chain OHLCV candles are sourced from Jupiter's public price API with a
  fallback to the latest spot quote when historical data is unavailable.
* **onchain_default_quote** – quote currency used when appending to entries in
  `onchain_symbols`. Defaults to `USDC`.
* **allow_short** – enable short selling. Set to `true` only when your exchange account supports short selling.
* Trades and swaps automatically retry transient errors up to three times.

### Market Scanning
* **symbol_batch_size** – number of symbols processed each cycle.
  The same batch size controls the initial market scan at startup where
  progress is logged after each batch.
* **scan_lookback_limit** – default maximum candles per timeframe fetched during
  the initial scan (`700`). `update_multi_tf_ohlcv_cache` trims this value for
  newly listed pairs using `get_kraken_listing_date` so requests never exceed the
  available history. The caches store at least this many bars when history
  permits.
* **scan_deep_top** – number of top-ranked pairs loaded with a full year of history during startup (default `50`).
* **start_since** – optional timestamp used to backfill older data during the initial scan.
  When set, the bot loads candles starting from this time (e.g. `365d` for one year)
  before switching to realtime updates.
* **bootstrap resume** – completed `symbol × timeframe` pairs are recorded in
  `cache/ohlcv_bootstrap_state.json` so restarts continue from the last success.
  Run `python -m crypto_bot.cli cache reset-bootstrap` to clear this file and
  force a fresh bootstrap.
* **min_history_fraction** – minimum portion of candles that must be retrieved
  for a pair to remain cached. Defaults to `0.5`.
* **cycle_lookback_limit** – candles fetched each cycle. Defaults to `150`.
* **adaptive_scan.enabled** – turn on dynamic sizing.
* **adaptive_scan.atr_baseline** – ATR level corresponding to a 1× factor.
* **adaptive_scan.max_factor** – cap multiplier for batch size and scan rate.
* **symbol_refresh_minutes** – minutes before the symbol queue is refreshed.
* **symbol_filter** - filters by minimum volume, 24h change percentile, spread and correlation.
* **skip_symbol_filters** – bypass the volume and spread checks and use the provided symbol list as-is.
* **symbol_score_weights** – weights for volume, change, spread, age, latency and liquidity. These defaults can be overridden in `config.yaml` and must sum to a positive value.

```yaml
symbol_score_weights:
  volume: 0.4
  change: 0.2
  spread: 0.2
  age: 0.1
  latency: 0.1
  liquidity: 0.0
```
* **uncached_volume_multiplier** – extra volume factor applied when a pair is missing from `cache/liquid_pairs.json`.
* If the cache file does not exist and the initial scan yields no symbols, the bot calls `tasks.refresh_pairs.refresh_pairs_async` to fetch a fresh list before aborting.
* **min_symbol_age_days** – skip newly listed pairs.
* **min_symbol_score** – minimum score required for trading.
* **top_n_symbols** – maximum number of active markets.
* **max_age_days**, **max_change_pct**, **max_spread_pct**, **max_latency_ms**, **max_vol** – additional scanning limits.
* **use_numba_scoring** – enable numba acceleration for symbol scoring when available.
* **arbitrage_enabled** – compare CEX and Solana DEX prices each cycle. See
  `scan_cex_arbitrage` in `crypto_bot/main.py` for multi-exchange support.
* **solana_scanner.enabled** – continuously discover new Solana pools (enabled by
  default in `config.example.yaml`).
* **solana_scanner.gecko_search** – query GeckoTerminal to verify volume for new Solana tokens.
* **gecko_limit** – maximum simultaneous requests to GeckoTerminal. Reduce this if you encounter HTTP 429 errors.
* **max_concurrent_tickers** – maximum simultaneous ticker requests.
* **ticker_rate_limit** – delay in milliseconds after each ticker API call.
* Solana tokens are filtered using symbol scoring; adjust `min_symbol_score` to control the threshold.

### Risk Parameters
* **risk** – default stop loss, take profit and drawdown limits. `min_volume` is set to `0.0001` to filter thin markets. The stop is 1.5× ATR and the take profit is 3× ATR by default.
* **trade_size_pct** – percent of capital used per trade.
* **win_rate_threshold**/**win_rate_boost_factor** – boost position sizing when a
  strategy's win rate exceeds this threshold.
* **max_open_trades** – maximum simultaneous open trades.
* **max_slippage_pct** – slippage threshold for the depth-based check.
* **liquidity_check**/**liquidity_depth** – verify order book depth.
* **max_liquidity_usage** – limit how much of the visible liquidity a single
  order may consume.
* **weight_liquidity** – scoring weight for available pool liquidity on Solana pairs.
* **volatility_filter** - skips trading when ATR is too low or funding exceeds `max_funding_rate`. The minimum ATR percent is `0.0005`.
* **sentiment_filter** - checks the Fear & Greed index and Twitter sentiment to avoid bearish markets.
* **sl_pct**/**tp_pct** – defaults for Solana scalper strategies.
* **mempool_monitor** – pause or reprice when Solana fees spike.
* **priority_fee_cap_micro_lamports** – abort scalper trades when priority fees exceed this.
* **min_cooldown** – minimum minutes between trades.
* **cycle_bias** – optional on-chain metrics to bias trades.
* **min_expected_value** – minimum expected value for a strategy based on
  historical stats. Defaults to `-1.0`, effectively disabling the expected value check.
* **default_expected_value** – fallback EV when no stats exist. When unset,
  the expected value check is skipped.
* **drawdown_penalty_coef** – weight applied to historical drawdown when
  scoring strategies.
* **safety** – kill switch thresholds and API error limits.
* **scoring** – windows and weights used to rank strategies.
* **exec** – advanced order execution settings.
* **exit_strategy** – default stop loss/take profit settings plus partial profit
  taking and trailing stops. The trailing stop follows price by 2% after at
  least 1% gain. `momentum_bot` uses the `trailing_stop_factor` setting for an
  ATR-based stop.

### Strategy and Signals
* **strategy_allocation** – capital split across strategies.
* **strategy_evaluation_mode** – how the router chooses a strategy.
* **ensemble_min_conf** – minimum confidence required for a strategy to
  participate in ensemble evaluation.
* **voting_strategies**/**min_agreeing_votes** – strategies used for the voting router.
* **micro_scalp_bot** – EMA settings plus volume z-score and ATR filters for the scalp bot.
  Supports tick-level aggregation, optional mempool fee checks and order-book
  imbalance filtering with an optional penalty. Set `trend_filter` or
  `imbalance_filter` to `false` to bypass the trend or order book checks.
* **pattern_timeframe** – optional candle interval used by the bounce scalper to
  confirm engulfing or hammer patterns. Defaults to `1m`.
* **force** – pass `force=True` to `bounce_scalper.generate_signal` to bypass the cooldown and win-rate filter for one bounce scalper cycle.
* **cooldown_enabled** – disable to ignore the cooldown and win-rate check.
* **breakout** – Bollinger/Keltner squeeze with `donchian_window`,
  `vol_confirmation`/`vol_multiplier`, `setup_window`, `trigger_window` and a
  `risk` section for stop sizing. Lowering `vol_multiplier` (e.g., `0.5`)
  captures more frequent breakouts on tokens like Solana. The optional
  `momentum_filter` is disabled by default for higher trade frequency.
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
size = risk_manager.position_size(
    score,
    balance,
    lower_df,
    atr=atr,
    name="trend_bot",
)
```
* Passing a ``name`` parameter causes ``position_size`` to check
  ``get_recent_win_rate``. When the win rate exceeds ``win_rate_threshold``
  the calculated size is multiplied by ``win_rate_boost_factor``.
* Specify ``direction="short"`` to return a negative size for short trades.
* **ml_signal_model**/**signal_weight_optimizer** – blend strategy scores with machine-learning predictions.
* **signal_threshold**, **min_confidence_score**, **min_consistent_agreement** – thresholds for entering a trade. `min_confidence_score` and `signal_fusion.min_confidence` default to `0.0001`.
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
  When the token registry monitors pump.fun or Raydium for new listings it
  now auto-snipes price discrepancies between Kraken and Coinbase. If both
  exchanges list the token and the spread exceeds `0.5%`, a cross-chain
  arbitrage helper swaps the asset into BTC and updates the wallet balance.
  The previous `cross_chain_arb_bot.py` module has been consolidated into
  helper functions. Use `scan_cex_arbitrage` or `scan_arbitrage` in
  `crypto_bot/main.py` when implementing a multi-exchange arbitrage strategy.
* **scoring_weights** - weighting factors for regime confidence, symbol score and volume metrics.
* **signal_fusion** – enable to combine scores from multiple strategies via a `fusion_method` for improved selection.
* **strategy_router** - maps market regimes to lists of strategy names. Each regime also accepts a `<regime>_timeframe` key (e.g. `trending_timeframe: 1h`, `volatile_timeframe: 1m`). The `range_arb_bot` strategy is always active and does not need to be listed in any regime.
* **mode_threshold**/**mode_degrade_window** - degrade to manual mode when auto selection underperforms.
* **meta_selector**/**rl_selector** – experimental strategy routers. Enable these along with `signal_fusion` for improved strategy selection.
  Train the meta selector on your historical trade results with:

  ```bash
  python crypto_bot/tools/train_meta_selector.py
  ```

  The script reads `crypto_bot/logs/strategy_performance.json` and saves the
  resulting LightGBM model to `crypto_bot/models/meta_selector_lgbm.pkl`.
* **bandit_router** – Thompson sampling router that favors historically profitable strategies.
* **bandit** – Thompson sampling selector; tune `explore_pct` for exploration.
* **mode** – set to `auto` so the router switches between CEX and on-chain
  strategies whenever a token's mint is known, or `manual` to lock in the
  configured exchange.
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

Passing `force=True` to `bounce_scalper.generate_signal()` bypasses the filter for a single cycle.

#### Mean Bot
The mean reversion bot now incorporates an ADX trend filter to avoid
counter‑trend trades. Its RSI thresholds are scaled according to recent
volatility, and you can optionally blend the final score with a machine
learning prediction. Enable the weighting in `crypto_bot/config.yaml`:

```yaml
mean_bot:
  ml_enabled: true
```
This requires the optional `coinTrader_Trainer` package and a trained model.
If either is missing, the bot logs a message and runs without machine learning
predictions. The bot only opens positions when the current
20-bar Bollinger bandwidth is below its 20-bar median, reducing trades during
ranging periods and improving the win rate.

#### DCA Bot
Dollar-cost averaging gradually accumulates a position by buying when the price
closes 10% below the 20-period moving average or selling when it trades 10%
above. Adjust the behaviour in
`crypto_bot/config.yaml`:

```yaml
dca_bot:
  enabled: true          # turn on the strategy
  interval_minutes: 60   # wait time between DCA orders
  max_entries: 5         # total DCA steps
  order_size_pct: 0.02   # fraction of balance per order
```

Repeatedly adding exposure can amplify losses during prolonged downtrends, so
keep the order size small and stop averaging if drawdown exceeds your risk
tolerance.

#### LSTM Bot
The LSTM bot forecasts short-term price movement using a recurrent neural
network. It opens trades in the predicted direction once the model
confidence exceeds a threshold. Configure the feature in
`crypto_bot/config.yaml`:

```yaml
lstm_bot:
  enabled: true             # activate the LSTM strategy
  forecast_horizon: 3       # bars to predict ahead
  confidence_threshold: 0.55
```

#### Momentum Bot
The momentum bot rides strong trends using an ATR-based trailing stop. Set
`momentum_bot.risk.trailing_stop_factor` in `crypto_bot/config.yaml` to control
the stop distance.

```yaml
# crypto_bot/config.yaml
momentum_bot:
  fast_length: 20
  slow_length: 50
  risk:
    trailing_stop_factor: 1.5  # ATR multiple
    risk_pct: 0.01
```

#### Cross-Chain Arbitrage Bot
`cross_chain_arb_bot` watches price differences between a centralized
exchange and Solana DEX quotes. When the gap for a configured `pair` exceeds
`spread_threshold` (fractional), the bot buys on the cheaper venue and sells on the
other, covering bridge costs in the process.
The optional `fee_threshold` key helps avoid trades during periods of high on-chain fees.

```yaml
cross_chain_arb_bot:
  pair: SOL/USDC
  spread_threshold: 0.05
  fee_threshold: 30
```


#### Flash Crash Scalper
The flash crash scalper enters after a rapid ~10% drop confirmed by a spike in
volume and exits once price rebounds about 5%.

```yaml
flash_crash_scalper:
  drop_threshold: 0.10      # enter after a 10% decline
  rebound_target: 0.05      # take profit on 5% bounce
  volume_multiplier: 2.0    # volume spike relative to average
```

### Data and Logging
* **timeframe** – base interval for most indicators (default `15m`).
* **timeframes** – list of additional intervals cached for reuse by strategies.
* **scalp_timeframe** – short interval used by the scalping bots.
* **ohlcv_snapshot_frequency_minutes**/**ohlcv_snapshot_limit** – OHLCV caching options. A separate cache is maintained for each timeframe listed in `timeframes`.
* **timeframe**, **timeframes**, **scalp_timeframe** – candle intervals used for analysis. Default `timeframe` is `15m` and `timeframes` include `1m`, `5m`, `15m`, and `1h` (Coinbase lacks a `4h` interval).
* **ohlcv_snapshot_frequency_minutes**/**ohlcv_snapshot_limit** – OHLCV caching options. The snapshot limit defaults to `500`.
* **loop_interval_minutes** – delay between trading cycles (default `0.05`). Shorter delays increase CPU usage as the loop runs more often.
* **ohlcv_timeout**, **max_concurrent_ohlcv**, **max_ohlcv_failures** – limits for candle requests.
* **ohlcv_batch_size** – number of symbols grouped per OHLCV request.
* **redis_url** – optional Redis connection string for caching OHLCV data.
* **ohlcv.storage_path** – directory used to persist OHLCV caches between runs.
* **tail_overlap_bars** – number of candles reloaded on startup to splice cached and new data.
* **max_bootstrap_bars** – cap on candles fetched per timeframe when no cache exists.
* **max_parallel** – number of markets processed concurrently.
* **gecko_limit** – concurrent GeckoTerminal requests.
* **max_concurrent_tickers** – maximum simultaneous ticker requests.
* **ticker_rate_limit** – delay in milliseconds after each ticker API call.
* **log_to_google** – export trades to Google Sheets.
* **quiet_mode** – suppress monitor output when stdout is not a TTY.
* **telegram** – bot token, chat ID and trade notifications. Optional
  **status_updates**, **bootstrap_updates** and **balance_updates** flags
  control startup messages, bootstrap progress and balance alerts.
* **telegram.message_interval** – minimum seconds between notifications.
* **telegram.max_messages_per_minute** – maximum Telegram sends per minute.
* **balance_change_threshold** – delta for Telegram balance alerts.
* **balance_poll_mod** – how often to poll balance between trades.
* **tax_tracking** – CSV export of executed trades.
* **metrics_enabled**, **metrics_backend**, **metrics_output_file** – cycle metrics output.
* **testing_mode** – indicates a sandbox environment.

Set `redis_url` to a standard Redis connection string (e.g. `redis://localhost:6379/0`) to cache recent OHLCV candles and reduce API calls.

#### Persistent OHLCV storage

When `ohlcv.storage_path` is set, the bot writes OHLCV data to disk so later runs can reuse it. The first run boots each symbol/timeframe by requesting up to `max_bootstrap_bars` candles and saving them under this directory. Subsequent runs load the cached files and request only new data, refetching the last `tail_overlap_bars` candles to avoid gaps.

Example first vs. second run output:

```text
2024-05-10 00:00:00 | INFO | market_loader | BTC/USD 1m bootstrap: fetching 1500 bars
2024-05-10 00:05:00 | INFO | market_loader | BTC/USD 1m: using cache, requesting 5 new bars
```

The second run reused nearly all local data and hit the exchange for only a handful of candles.

### Kraken Call Rate Limits

Kraken allows roughly **1 request per second** with a short burst capacity of
about 20 calls every 10 seconds. The bot enforces this using an internal
`Semaphore` to gate outgoing requests and a small backoff that increases the
delay when a call fails. Tune these options in `crypto_bot/config.yaml` to stay
within the limit:

* **max_concurrent_tickers** – maximum ticker requests processed in parallel.
* **max_concurrent_ohlcv** – maximum concurrent OHLCV fetches.
* **ohlcv_batch_size** – number of symbols loaded per OHLCV batch.
* **ticker_backoff_initial** – seconds to wait after a ticker failure.
* **ticker_backoff_max** – maximum delay between ticker retries.

Start with low values (e.g. 2 tickers and 4 OHLCV requests) and raise them only
if you remain under the limit. If you notice log messages like
`Fetching for 1 symbols` repeating, the semaphore is throttling requests. Reduce
these settings or the batch size so each cycle completes without hitting the
cap.
## Exchange Setup for U.S. Users


1. Create API keys on **Coinbase Advanced Trade** or **Kraken**.
2. (Optional) Run `python crypto_bot/wallet_manager.py` to generate `user_config.yaml`. Any
   credentials found in the environment will be used automatically. `python -m crypto_bot.main` will start the same wizard when credentials or `user_config.yaml` are missing.
3. Place your API keys in `.env` as shown in the configuration
   section above. Environment variables take precedence over values stored in
   `user_config.yaml`.
3. Fill out `.env` with your API keys and optional `FERNET_KEY`.
   Environment variables take precedence over values stored in
   `user_config.yaml`. If you prefer to enter credentials interactively,
   leave the entries commented out.

   ```env
   # EXCHANGE=coinbase  # or kraken
   # API_KEY=your_key
   # API_SECRET=your_secret
   # API_PASSPHRASE=your_coinbase_passphrase_if_needed
   # COINBASE_API_KEY=your_coinbase_key
   # COINBASE_API_SECRET=your_coinbase_secret
   # COINBASE_API_PASSPHRASE=your_coinbase_passphrase
   # KRAKEN_API_KEY=your_kraken_key
   # KRAKEN_API_SECRET=your_kraken_secret
    # SUPABASE_URL=your_supabase_project_url
    # SUPABASE_SERVICE_ROLE_KEY=your_service_role_key  # or SUPABASE_KEY/SUPABASE_API_KEY
   # FERNET_KEY=your_generated_fernet_key
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
     message_interval: 1.0
     max_messages_per_minute: 20
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
   - `/panic_sell` – exit all open positions immediately (paper or live).
      Liquidations are recorded in `crypto_bot/logs/positions.log` while
      overall activity continues in `bot.log`. After closing a position you can
      confirm the updated wallet total by reading the last "Wallet balance"
      entry in this file.
   - HTTP `POST /close-all` – trigger the same exit via the web server,
      also works in paper trading mode.
4. If you see `Failed to send message: Not Found` in the logs, the chat ID or
   token is likely incorrect or the bot lacks permission to message the chat.
   Double-check the values in `config.yaml` and ensure you've started a
  conversation with your bot.

For asynchronous workflows the notifier exposes `notify_async` so you can
send messages without blocking the event loop:

```python
notifier = TelegramNotifier.from_config(cfg["telegram"])

async def heartbeat():
    while True:
        await notifier.notify_async("Bot running")
        await asyncio.sleep(60)
```

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

Add `TWITTER_SENTIMENT_URL` to `.env` to point at the sentiment
service used by `sentiment_filter.py`. If this variable is not provided, the bot
defaults to the placeholder `https://api.example.com/twitter-sentiment`, so
sentiment fetches will fail until a real URL is supplied.

### Funding Rate API

Add `FUNDING_RATE_URL` to `.env` to specify the endpoint used by
`volatility_filter.py` when fetching perpetual funding rates. Without this
variable the bot falls back to the placeholder `https://funding.example.com`
and will log errors until a real URL is supplied.

For Kraken, add the following line to `.env`:

```env
FUNDING_RATE_URL=https://futures.kraken.com/derivatives/api/v3/historical-funding-rates?symbol=
```

`volatility_filter.py` will append the instrument symbol directly to this URL
when requesting funding information.

### LunarCrush API

Sign up at [LunarCrush](https://lunarcrush.com/developers) to generate an API
key. Set `LUNARCRUSH_API_KEY` in `.env` or export it in your
environment so the bot can retrieve social metrics. The free tier allows about
100 requests per day.


4. In `crypto_bot/config.yaml` set:

   ```yaml
   exchanges: [coinbase, kraken]
   primary_exchange: coinbase
   secondary_exchange: kraken
   execution_mode: dry_run  # or live
   use_websocket: true      # enable when trading on Kraken via WebSocket
   ```

For Kraken, a WebSocket token is requested automatically using your API key and secret. To override the fetched token, add it to `.env`:

```env
# KRAKEN_WS_TOKEN=your_ws_token  # optional override
```

5. In `crypto_bot/config.yaml` set:

    ```yaml
    exchanges: [coinbase, kraken]
    primary_exchange: coinbase
    secondary_exchange: kraken
    execution_mode: dry_run  # or live
    use_websocket: true      # enable when trading on Kraken via WebSocket
    ```

Additional execution flags:

```yaml
liquidity_check: true        # verify order book liquidity before placing orders
liquidity_depth: 10          # order book depth levels to inspect
max_liquidity_usage: 0.8     # cap order size to this fraction of depth
weight_liquidity: 0.0        # symbol score weight for pool liquidity
twap_enabled: false          # split large orders into slices
twap_slices: 4               # number of slices when TWAP is enabled
twap_interval_seconds: 10    # delay between TWAP slices
timeframe: 15m               # candles for regime detection
scalp_timeframe: 1m          # candles for micro_scalp_bot/bounce_scalper
loop_interval_minutes: 0.05  # wait time between trading cycles
force_websocket_history: false  # set true to disable REST fallback
max_ws_limit: 50             # skip WebSocket when request exceeds this
gecko_limit: 10              # concurrent GeckoTerminal requests
ohlcv_timeout: 60             # request timeout for OHLCV fetches
max_concurrent_ohlcv: 4      # limit simultaneous OHLCV fetches
ohlcv_batch_size: 10         # group symbols per OHLCV fetch
max_concurrent_tickers: 20   # limit simultaneous ticker fetches
ticker_rate_limit: 0         # override exchange rate limit (ms)
poll_timeout: 60             # seconds to wait for order fill or transaction confirmation
force_websocket_history: true  # set false to enable REST fallback
max_ws_limit: 200            # skip WebSocket when request exceeds this
gecko_limit: 10              # concurrent GeckoTerminal requests
ohlcv_timeout: 60             # request timeout for OHLCV fetches
max_concurrent_ohlcv: 20     # limit simultaneous OHLCV fetches
ohlcv_batch_size: 10         # group symbols per OHLCV fetch
metrics:
  enabled: true              # write cycle statistics to metrics.csv
  file: crypto_bot/logs/metrics.csv
```

`loop_interval_minutes` determines how long the bot sleeps between each
evaluation cycle. The default of `0.05` minutes (~3 seconds) balances
responsiveness with CPU usage. Shortening this delay can noticeably increase
CPU load.
`max_concurrent_ohlcv` caps how many OHLCV requests run in parallel when
`update_ohlcv_cache` gathers new candles. The new `ohlcv_timeout` option
controls the timeout for each fetch call. If you still encounter timeouts after
raising this value, try lowering `max_concurrent_ohlcv` or `ohlcv_batch_size` to reduce pressure on
the exchange API.
Trade execution helpers now poll for order status or transaction confirmation for up to `poll_timeout` seconds (default 60) and gracefully handle partial fills.
Orders now compute slippage using the specified `liquidity_depth`. When the
calculated slippage exceeds `max_slippage_pct` the bot automatically switches to
TWAP execution, dividing the trade into slices so each consumes at most
`max_liquidity_usage` of the visible depth.
The updater automatically determines how many candles are missing from the
cache, so even when `limit` is large it only requests the data required to fill
the gap, avoiding needless delays.
The bot caches the last candle timestamp for open positions and skips updating
their history until a new bar appears.
The `metrics` section enables recording of cycle summaries to the specified CSV
file for later analysis.
`scalp_timeframe` sets the candle interval specifically for the micro_scalp_bot
and bounce_scalper strategies while `timeframe` covers all other analysis.

When `use_websocket` is enabled the bot uses a dedicated Kraken WebSocket
client for realtime streaming data. Disable websockets if you do not need
this functionality. Private order feeds are disabled by default; set
`kraken.use_private_ws` to `true` to enable them. When `false`, trading
continues via REST only.
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
regardless of the WebSocket setting. It relies on
`update_multi_tf_ohlcv_cache` to pull each timeframe. When `start_since` is
omitted the helper `get_kraken_listing_date` checks when every pair was
listed on Kraken and caps the request so no more candles are fetched than the
exchange actually offers. When sufficient history is available the loader
retrieves up to `scan_lookback_limit` candles per pair (700 by default on
Kraken) before switching to WebSocket updates.

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

When the `symbols` list is empty, the bot loads all active Kraken trading pairs
at startup by default. Pairs listed under `excluded_symbols` are skipped.
Disable this behaviour by setting `scan_markets` to `false`. When
`scan_in_background` is `true` the scan runs as
a background task so trading phases start immediately. Set it to `false` to
wait for scanning to complete before trading begins.

```yaml
scan_markets: true    # default
scan_in_background: true
symbols: []            # automatically populated
onchain_default_quote: USDT
onchain_symbols: ["SOL", "BONK", "AI16Z"]             # quote appended automatically
                                                # to mints via Jupiter/Helius
                                                # base must be mint if unknown
excluded_symbols: [ETH/USD]
exchange_market_types: ["spot"]  # options: spot, margin, futures
min_symbol_age_days: 2           # skip pairs with less history
symbol_batch_size: 50            # symbols processed per cycle
scan_lookback_limit: 700         # max candles per pair during startup
                                 # trimmed using Kraken listing data
scan_deep_top: 50                # deep load this many ranked symbols
start_since: 365d                # backfill candles this far in the past
min_history_fraction: 0.5        # minimum portion of history required
cycle_lookback_limit: 150        # candles fetched each cycle
max_spread_pct: 4.0              # skip pairs with wide spreads
```

To avoid loading every market on startup, populate `symbols` with the
top 200 pairs by volume from `tasks/refresh_pairs.py`. Set
`scan_markets: false` when you don't want to evaluate the entire exchange.

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

Some markets are synthetic index pairs (e.g., `AIBTC/USD`) or are known to
fail consistently when requesting historical data. These are filtered out
during symbol discovery using simple heuristics, and the loader logs a
summary such as `Excluded {n} synthetic/index pairs`. Additional markets
may be listed in `UNSUPPORTED_SYMBOLS` within
`crypto_bot/utils/market_loader.py` to skip them without making network
requests.

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
  min_volume_usd: 500
  volume_percentile: 5           # keep pairs above this volume percentile
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
max_concurrent_ohlcv: 2          # simultaneous OHLCV requests during startup
  max_concurrent_tickers: 20       # simultaneous ticker requests
  ticker_rate_limit: 0             # ms delay after each ticker request
  ticker_backoff_initial: 2        # seconds after first failure
  ticker_backoff_max: 60           # cap for exponential backoff
  initial_timeframes: [1m, 5m, 15m, 1h]  # preloaded intervals (4h unsupported on Coinbase)
initial_history_candles: 700     # candles fetched per timeframe initially
```

For thin markets you may want to relax the filters and trading
thresholds.  Lowering `min_volume_usd`, widening `max_spread_pct` and
reducing the minimum signal requirements lets the bot operate on less
liquid pairs:

```yaml
min_confidence_score: 0.00005
signal_threshold: 0.002
symbol_filter:
  max_spread_pct: 8
  min_volume_usd: 20
```

`max_concurrent_ohlcv` controls how many OHLCV requests are made in parallel
during the startup scan. It defaults to `2`, keeping API usage modest.
`initial_timeframes` lists the intervals preloaded before trading begins. When
omitted it falls back to the `timeframes` list (1m, 5m, 15m and 1h by
default; Coinbase does not offer 4h candles). `initial_history_candles` sets how many bars to download for each of
these intervals during the initial scan and defaults to the
`scan_lookback_limit` of `700`. Together these options ensure enough historical
data is fetched for regime detection and correlation checks before live trading
starts.
  max_concurrent_ohlcv: 10       # limit OHLCV requests when loading history
  initial_timeframes: [1h, 1d]  # timeframes fetched for new symbols
  initial_history_candles: 300   # candles per timeframe on first load
```

* **max_concurrent_ohlcv** – cap simultaneous OHLCV requests while scoring new symbols (default `10`).
* **ohlcv_batch_size** – batch size for grouped OHLCV requests.
* **max_concurrent_tickers** – cap simultaneous ticker requests (default `20`).
* **ticker_rate_limit** – delay applied after ticker requests in milliseconds.
* **initial_timeframes** – candle intervals pulled when caching a new market (default `[1h, 1d]`; Coinbase does not provide 4h candles).
* **initial_history_candles** – number of candles per timeframe loaded on first
  use (default `300`). The loader observes Kraken listing dates so it never
  requests data preceding a pair's debut.

Kraken labels Bitcoin as `XBT` in its market identifiers. The bot
automatically converts canonical symbols using `exchange.market_id`,
so configuration files may still list pairs like `BTC/USDT`.

`setup_window` controls how many candles of ticker history are gathered before
a symbol is eligible to trade, while `trigger_window` defines the period after
a setup is detected during which entry conditions must appear.  Each strategy
can also include a `risk` section such as `max_concurrent` or
`daily_loss_cap` to cap simultaneous positions and total daily losses.
WebSocket streaming is enabled by default when scanning. Set
`use_websocket: false` to force REST polling instead. You can disable
WebSocket just for ticker scanning by adding
`exchange.options.ws_scan: false` to your configuration while leaving
`use_websocket: true` for trading. If the WebSocket connection closes with
code `1006` the bot will automatically disable `ws_scan` and fall back to
HTTP. To re-enable WebSocket scanning set `exchange.options.ws_scan: true`
again in your configuration. When using the REST fallback the bot
requests tickers in batches controlled by
`symbol_filter.kraken_batch_size` to avoid Kraken's occasional `520`
errors. The public `/Ticker` calls also obey
`symbol_filter.http_timeout`.
The internal `_refresh_tickers` helper now skips any symbols missing from
`exchange.markets` to avoid unnecessary fetch attempts.
It also warns and skips caching when the ticker API returns errors or empty
results so zero-volume entries don't pollute the liquidity cache.
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
  min_quote_volume_usd: 10000000
  refresh_interval: 6h
  top_k: 40
  secondary_exchange: coinbase
  allowed_quote_currencies: []
  blacklist_assets: []
```
These thresholds can be changed in `config.yaml`.
Run it manually whenever needed:

```bash
python tasks/refresh_pairs.py --once
```
Run the script from the project root so that `cache/liquid_pairs.json` is
written where the bot expects it. Executing the command from another
directory may create a separate `cache` folder and lead to missing symbols.
Removing the `--once` flag keeps it running on the configured interval.
To ensure `cache/liquid_pairs.json` stays current you should schedule this
command to run automatically. One option is cron:

```cron
0 * * * * cd /path/to/coinTrader2.0 && /usr/bin/python3 tasks/refresh_pairs.py
```
You can accomplish the same thing with a systemd timer. Create the following
service and timer units:

```ini
# /etc/systemd/system/refresh_pairs.service
[Unit]
Description=Update trading pair cache

[Service]
Type=oneshot
WorkingDirectory=/path/to/coinTrader2.0
ExecStart=/usr/bin/python3 tasks/refresh_pairs.py --once

# /etc/systemd/system/refresh_pairs.timer
[Unit]
Description=Run refresh_pairs hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Enable it with `sudo systemctl enable --now refresh_pairs.timer`.

Delete `cache/liquid_pairs.json` to force a full rebuild on the next run.
If you see warnings about unsupported markets in the logs, regenerate the file
with `tasks/refresh_pairs.py --once`.

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

Set `FLASK_DEBUG=1` before running the command to enable Flask debug mode.

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
  decision messages. `[EVAL]` lines record each symbol's evaluation outcome in
  this file.
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
- `tools/split_log_levels.py` – split `bot.log` into separate files for
  INFO and WARNING messages. Run `python tools/split_log_levels.py` to
  produce `info.log` and `warning.log`.

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
If `priority_fee_cap_micro_lamports` is set, the scalper aborts the swap entirely when
the priority fee exceeds this limit.

## Solana Meme-Wave Sniper

This module watches for new liquidity pools on Solana and attempts to buy
into meme tokens before the crowd. Events from a Helius endpoint are
filtered through safety checks, scored, and executed using Jupiter quotes
bundled via Jito. A Twitter sentiment score can boost the ranking when the
tweet volume is high.

### Configuration

Add a `meme_wave_sniper` section to `crypto_bot/config.yaml`:

Set `HELIUS_API_KEY` in `.env` or as an environment variable. The pool
URL should reference this key so Helius can authorize the requests:

```yaml
meme_wave_sniper:
  enabled: true
  pool:
    url: https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_API_KEY}
    interval: 5
    websocket_url: wss://atlas-mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}
    raydium_program_id: EhhTK0i58FmSPrbr30Y8wVDDDeWGPAHDq6vNru6wUATk
  scoring:
    weight_liquidity: 1.0
    weight_tx: 1.0
    weight_social: 0.5
  safety:
    min_liquidity: 50
  risk:
    max_concurrent: 20
    daily_loss_cap: 1.5
  execution:
    dry_run: true

```
Set the `HELIUS_API_KEY` environment variable with your Helius API key.

Add a `yamlmeme_sniper` section for probability-based sniping:

```yaml
yamlmeme_sniper:
  min_pump_prob: 0.6  # Assessment threshold
  max_dev_hold_pct: 20
  min_liquidity_usd: 5000
  twitter_sentiment_min: 70
  helius_api_key: your_key_here  # For metadata
```

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

### Meme Wave Bot

The meme wave bot trades sudden hype cycles discovered on centralized
exchanges and in Solana mempools. It monitors for sharp volume spikes,
mempool activity and optional Twitter excitement before buying. Most
profits will likely arise from Solana where new tokens see the fastest
hype waves.
Add a `meme_wave_bot` section to `crypto_bot/config.yaml`:

```yaml
meme_wave_bot:
  vol_spike_thr: 2.0        # volume spike multiple triggering the bot
  sentiment_thr: 0.6        # minimum sentiment score on Twitter
  twitter_query: null       # optional query for the sentiment API
```

- **vol_spike_thr** – ratio of current volume to the recent average required
  to consider entering a trade.
- **sentiment_thr** – threshold for the normalized Twitter sentiment score
  before a meme wave trade is executed.
- **twitter_query** – search term used when fetching sentiment. If omitted the
  trading symbol is used.

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

## Meme Wave Rider

This scalper rides trending meme coins by watching mempool volume alongside
Twitter sentiment. A rolling volume window flags spikes in Solana swaps while a
sentiment API call checks for positive chatter. When both metrics exceed your
thresholds, a trade is executed using `solana_trading.py`.

### Configuration

Add a `meme_wave_bot` section to `crypto_bot/config.yaml`:

```yaml
meme_wave_bot:
  enabled: true
  mempool_window: 60        # seconds to track volume
  volume_threshold: 5000    # SOL per minute
  sentiment_threshold: 0.7  # positive ratio
```

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

## ML Regime Trainer

Use the optional `cointrader-trainer` package only when you need to train and
upload new models for the regime classifier. After generating trade logs, run
the trainer to upload a LightGBM model to Supabase:

```bash
python ml_trainer.py train regime --use-gpu --federated
```

Ensure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_KEY`, or `SUPABASE_API_KEY` are set
in `.env` so the upload succeeds. Set
`use_ml_regime_classifier: true` in `crypto_bot/regime/regime_config.yaml` to
enable automatic downloads of the trained model when the bot starts.

### Ingest a 7-column minute CSV (CSV7) and train locally

Our CSV7 format is headerless with 7 columns: `ts,open,high,low,close,volume,trades`.

**Ingest → normalized CSV (and Parquet if available):**
```bash
cointrainer import-csv7 --file ./XRPUSD_1.csv --symbol XRPUSD --out ./data/XRPUSD_1m
```

### Chunked imports and resume

`import-csv7` can process large files in smaller pieces. Control how many rows
are handled at a time with the new `chunk_size` option. After each chunk a
progress line is printed and appended to the resume file in the format
`<start_ts>,<end_ts>,<rows_written>`. Re-running the command reads that file and
skips any ranges that were already completed so interrupted runs pick up where
they left off. Delete the resume file to reset the mechanism and start again
from the beginning.

Example configuration:

```yaml
file: ./XRPUSD_1.csv
symbol: XRPUSD
out: ./data/XRPUSD_1m
chunk_size: 10000          # rows per chunk
resume: true               # enable resume using the progress file
resume_file: ./data/XRPUSD_1m.progress
```

## Development Setup

Use Python&nbsp;3.10 or later to avoid package incompatibilities. Set up a
virtual environment and install the dependencies:

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   ```

2. **Install dependencies**

   `requirements.txt` references the optional `cointrader-trainer` package. It
   is only needed if you plan to train new models; runtime model downloads from
   Supabase do not require it. If you wish to train models, install it from its
   source or uncomment the line before running the command below:

   ```bash
   pip install -r requirements.txt
   ```

   Install the additional development packages if you plan to run the test
   suite:

   ```bash
   pip install -r requirements-dev.txt
   ```

The project still relies on Pydantic&nbsp;1.x. If you see errors related to
Pydantic&nbsp;2, ensure the dependency is pinned with `pydantic<2`.

This project pins **Pydantic** to version `1.10.22` in `requirements.txt` because
FastAPI 0.97.0 still depends on the 1.x API. Using the latest 1.x release avoids
compatibility issues introduced in Pydantic 2.x while keeping bug fixes from the
final 1.x series.

Run the test suite to verify your environment:

```bash
pytest -q
```

## Testing

The repository includes an automated test suite. Some tests rely on optional
packages such as `numpy`, `pandas`, `pytest-asyncio`, `ccxt`, `flask`, `base58`,
`prometheus_client`, `python-dotenv` and `websocket-client`. The **full** set of
tests requires the dependencies listed in `requirements.txt` together with the
additional packages enumerated in `requirements-dev.txt`. When running only a
subset of tests you may need to install these optional packages manually so
that `pytest` can import the modules they depend on.

Set up the environment by running the provided script:

```bash
bash codex/setup.sh  # installs system and Python dependencies
```

Alternatively you can install the Python packages manually:

```bash
pip install -r requirements.txt       # core dependencies
pip install -r requirements-dev.txt   # optional packages for tests
```

Run `pip install -r requirements-dev.txt` to ensure packages like `numpy`,
`pandas` and `pytest-asyncio` are installed before executing `pytest`.

If `pytest` fails with a `ModuleNotFoundError`, ensure the packages from both
requirements files are installed.  After the dependencies are available, execute

```bash
pytest -q
```

## Troubleshooting

* **Pydantic version errors** – The project has not yet been migrated to
  Pydantic&nbsp;2. If you encounter import issues, explicitly install
  `pydantic<2`.

High `max_concurrent_ohlcv` values combined with short `ohlcv_timeout`
settings can overload the exchange and lead to failed candle fetches.
Increase `ohlcv_timeout` to give each request more time and lower
`max_concurrent_ohlcv` if errors continue.
`gecko_request` retries failures using exponential backoff (1s, 2s, 4s) before giving up.

This project is provided for educational purposes only. Use it at your own risk, and remember that nothing here constitutes financial advice.


## License

This project is licensed under the [MIT License](LICENSE).

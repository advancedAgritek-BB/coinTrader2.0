import time
import pandas as pd
import ccxt
import yaml
from dotenv import dotenv_values
from pathlib import Path
import json

from crypto_bot.wallet_manager import load_or_create
from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import route
from crypto_bot.signals.signal_scoring import evaluate
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.execution.cex_executor import execute_trade as cex_trade, load_exchange
from crypto_bot.execution.solana_executor import execute_swap

CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'
ENV_PATH = Path(__file__).resolve().parent / '.env'


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    user = load_or_create()
    secrets = dotenv_values(ENV_PATH)
    exchange = load_exchange(secrets['API_KEY'], secrets['API_SECRET'])
    risk_config = RiskConfig(**config['risk'])
    risk_manager = RiskManager(risk_config)

    stats_file = Path('crypto_bot/logs/strategy_stats.json')
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    mode = user.get('mode', config['mode'])

    while True:
        ohlcv = exchange.fetch_ohlcv(config['symbol'], timeframe=config['timeframe'], limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if not risk_manager.allow_trade(df):
            time.sleep(config['loop_interval_minutes'] * 60)
            continue

        regime = classify_regime(df)
        env = mode if mode != 'auto' else 'cex'
        strategy_fn = route(regime, env)
        score, direction = evaluate(strategy_fn, df)

        if score < config['signal_threshold'] or direction == 'none':
            time.sleep(config['loop_interval_minutes'] * 60)
            continue

        balance = exchange.fetch_balance()['USDT']['free'] if config['execution_mode'] != 'dry_run' else 1000
        size = balance * config['trade_size_pct']

        if env == 'onchain':
            execute_swap('SOL', 'USDC', size, user['telegram_token'], user['telegram_chat_id'], dry_run=config['execution_mode'] == 'dry_run')
        else:
            cex_trade(
                exchange,
                config['symbol'],
                direction,
                size,
                user['telegram_token'],
                user['telegram_chat_id'],
                dry_run=config['execution_mode'] == 'dry_run',
            )

        key = f"{env}_{regime}"
        stats.setdefault(key, {'trades': 0})
        stats[key]['trades'] += 1
        stats_file.write_text(json.dumps(stats))

        time.sleep(config['loop_interval_minutes'] * 60)


if __name__ == '__main__':
    main()

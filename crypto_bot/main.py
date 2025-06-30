import time
import pandas as pd
import ccxt
import yaml
from dotenv import dotenv_values
from pathlib import Path

from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy.router import route
from crypto_bot.signals.signal_scoring import evaluate
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.execution.executor import execute_trade, load_exchange


CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'
ENV_PATH = Path(__file__).resolve().parent / '.env'


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    secrets = dotenv_values(ENV_PATH)
    exchange = load_exchange(secrets['API_KEY'], secrets['API_SECRET'])
    risk_config = RiskConfig(**config['risk'])
    risk_manager = RiskManager(risk_config)

    while True:
        ohlcv = exchange.fetch_ohlcv(config['symbol'], timeframe=config['timeframe'], limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        regime = classify_regime(df)
        strategy_fn = route(regime)
        score, direction = evaluate(strategy_fn, df)
        balance = exchange.fetch_balance()['USDT']['free'] if config['mode'] != 'dry_run' else 1000
        size = risk_manager.position_size(score, balance)
        if direction != 'none' and score > 0:
            execute_trade(
                exchange,
                config['symbol'],
                direction,
                size,
                config,
                secrets['TELEGRAM_TOKEN'],
                config['telegram']['chat_id'],
                dry_run=config['mode'] == 'dry_run'
            )
        time.sleep(config['loop_interval_minutes'] * 60)


if __name__ == '__main__':
    main()

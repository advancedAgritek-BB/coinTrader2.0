import os
import asyncio
import pandas as pd
import yaml
import asyncio
from dotenv import dotenv_values
from pathlib import Path
import json
from crypto_bot.utils.telegram import send_message
from crypto_bot.utils.logger import setup_logger

from crypto_bot.wallet_manager import load_or_create
from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import route
from crypto_bot.signals.signal_scoring import evaluate, evaluate_async
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.risk.exit_manager import (
    calculate_trailing_stop,
    should_exit,
    get_partial_exit_percent,
)

from crypto_bot.execution.cex_executor import (
    execute_trade as cex_trade,
    get_exchange,
    place_stop_order,
    execute_trade_async as cex_trade_async,
from crypto_bot.execution.cex_executor import execute_trade_async as cex_trade_async, get_exchange
from crypto_bot.execution.cex_executor import (
    execute_trade as cex_trade,
    get_exchange,
)
from crypto_bot.execution.solana_executor import execute_swap
from crypto_bot.fund_manager import (
    check_wallet_balances,
    detect_non_trade_tokens,
    auto_convert_funds,
)

CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'
ENV_PATH = Path(__file__).resolve().parent / '.env'

logger = setup_logger('bot', 'crypto_bot/logs/bot.log')


def load_config() -> dict:
    """Load YAML configuration for the bot."""

    with open(CONFIG_PATH) as f:
        logger.info("Loading config from %s", CONFIG_PATH)
        return yaml.safe_load(f)


async def main() -> None:
async def main():
def main() -> None:
    """Entry point for running the trading bot."""

    logger.info("Starting bot")
    config = load_config()
    secrets = dotenv_values(ENV_PATH)
    os.environ.update(secrets)
    user = load_or_create()
    exchange, ws_client = get_exchange(config)
    try:
        await asyncio.to_thread(exchange.fetch_balance)
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            await exchange.fetch_balance()
        else:
            await asyncio.to_thread(exchange.fetch_balance)
    except Exception as e:
        logger.error("Exchange API setup failed: %s", e)
        send_message(
            secrets.get('TELEGRAM_TOKEN'),
            config['telegram']['chat_id'],
            f"API error: {e}",
        )
        return
    risk_config = RiskConfig(**config['risk'])
    risk_manager = RiskManager(risk_config)

    open_side = None
    entry_price = None
    trailing_stop = 0.0
    position_size = 0.0
    highest_price = 0.0
    stats_file = Path('crypto_bot/logs/strategy_stats.json')
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    mode = user.get('mode', config['mode'])

    while True:
        # Detect deposits of BTC/ETH/XRP and convert to a tradeable token
        balances = await asyncio.to_thread(check_wallet_balances, user.get('wallet_address', ''))
        for token in detect_non_trade_tokens(balances):
            amount = balances[token]
            logger.info("Converting %s %s to USDC", amount, token)
            asyncio.run(
                auto_convert_funds(
                    user.get('wallet_address', ''),
                    token,
                    'USDC',
                    amount,
                    dry_run=config['execution_mode'] == 'dry_run',
                )
            await asyncio.to_thread(
                auto_convert_funds,
                user.get('wallet_address', ''),
                token,
                'USDC',
                amount,
                dry_run=config['execution_mode'] == 'dry_run',
                slippage_bps=config.get('solana_slippage_bps', 50),
            )

        ohlcv = await asyncio.to_thread(
            exchange.fetch_ohlcv,
            config['symbol'],
            timeframe=config['timeframe'],
            limit=100,
        )
        if config.get('use_websocket', False) and hasattr(exchange, 'watch_ohlcv'):
            ohlcv = await exchange.watch_ohlcv(config['symbol'], timeframe=config['timeframe'], limit=100)
        else:
            if asyncio.iscoroutinefunction(getattr(exchange, 'fetch_ohlcv', None)):
                ohlcv = await exchange.fetch_ohlcv(config['symbol'], timeframe=config['timeframe'], limit=100)
            else:
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv,
                    config['symbol'],
                    timeframe=config['timeframe'],
                    limit=100,
                )
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ohlcv = exchange.fetch_ohlcv(
            config['symbol'], timeframe=config['timeframe'], limit=100
        )
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )

        if not risk_manager.allow_trade(df):
            await asyncio.sleep(config['loop_interval_minutes'] * 60)
            continue

        regime = classify_regime(df)
        logger.info("Market regime classified as %s", regime)
        env = mode if mode != 'auto' else 'cex'
        strategy_fn = route(regime, env)
        score, direction = await evaluate_async(strategy_fn, df)
        logger.info("Signal score %.2f direction %s", score, direction)
        balance = (
            (await asyncio.to_thread(exchange.fetch_balance))["USDT"]["free"]
            if config["execution_mode"] != "dry_run"
        if config['execution_mode'] != 'dry_run':
            if asyncio.iscoroutinefunction(getattr(exchange, 'fetch_balance', None)):
                balance = (await exchange.fetch_balance())['USDT']['free']
            else:
                balance = (await asyncio.to_thread(exchange.fetch_balance))['USDT']['free']
        else:
            balance = 1000
        balance = (
            exchange.fetch_balance()['USDT']['free']
            if config['execution_mode'] != 'dry_run'
            else 1000
        )
        size = risk_manager.position_size(score, balance)

        current_price = df['close'].iloc[-1]

        if open_side:
            pnl_pct = ((current_price - entry_price) / entry_price) * (
                1 if open_side == 'buy' else -1
            )
            pnl_pct = (
                (current_price - entry_price) / entry_price
            ) * (1 if open_side == 'buy' else -1)
            if pnl_pct >= config['exit_strategy']['min_gain_to_trail']:
                if current_price > highest_price:
                    highest_price = current_price
                if highest_price:
                    trailing_stop = calculate_trailing_stop(
                        pd.Series([highest_price]),
                        config['exit_strategy']['trailing_stop_pct'],
                    )

            exit_signal, trailing_stop = should_exit(
                df, current_price, trailing_stop, config
                df,
                current_price,
                trailing_stop,
                config,
            )
            if exit_signal:
                pct = get_partial_exit_percent(pnl_pct * 100)
                sell_amount = (
                    position_size * (pct / 100)
                    if config['exit_strategy']['scale_out'] and pct > 0
                    else position_size
                )
                logger.info("Executing exit trade amount %.4f", sell_amount)
                await cex_trade_async(
                    exchange,
                    ws_client,
                    config['symbol'],
                    'sell' if open_side == 'buy' else 'buy',
                    sell_amount,
                    secrets['TELEGRAM_TOKEN'],
                    config['telegram']['chat_id'],
                    dry_run=config['execution_mode'] == 'dry_run',
                    use_websocket=config.get('use_websocket', False),
                    config=config,
                )
                if sell_amount >= position_size:
                    risk_manager.cancel_stop_order(exchange)
                    open_side = None
                    entry_price = None
                    position_size = 0.0
                    trailing_stop = 0.0
                    highest_price = 0.0
                else:
                    position_size -= sell_amount
                    risk_manager.update_stop_order(position_size)

        if score < config['signal_threshold'] or direction == 'none':
            await asyncio.sleep(config['loop_interval_minutes'] * 60)
            continue

        balance = (
            (await asyncio.to_thread(exchange.fetch_balance))["USDT"]["free"]
        if config['execution_mode'] != 'dry_run':
            if asyncio.iscoroutinefunction(getattr(exchange, 'fetch_balance', None)):
                balance = (await exchange.fetch_balance())['USDT']['free']
            else:
                balance = (await asyncio.to_thread(exchange.fetch_balance))['USDT']['free']
        else:
            balance = 1000
        size = balance * config['trade_size_pct']

        if env == 'onchain':
            asyncio.run(
                execute_swap(
                    'SOL',
                    'USDC',
                    size,
                    user['telegram_token'],
                    user['telegram_chat_id'],
                    dry_run=config['execution_mode'] == 'dry_run',
                )
            await asyncio.to_thread(
                execute_swap,
        balance = (
            exchange.fetch_balance()['USDT']['free']
            if config['execution_mode'] != 'dry_run'
            else 1000
        )
        size = balance * config['trade_size_pct']

        if env == 'onchain':
            await asyncio.to_thread(
                execute_swap,
            execute_swap(
                'SOL',
                'USDC',
                size,
                user['telegram_token'],
                user['telegram_chat_id'],
                dry_run=config['execution_mode'] == 'dry_run',
            )
            risk_manager.register_stop_order(
                {
                    'token_in': 'SOL',
                    'token_out': 'USDC',
                    'amount': size,
                    'dry_run': config['execution_mode'] == 'dry_run',
                }
            )
                slippage_bps=config.get('solana_slippage_bps', 50),
                dry_run=config['execution_mode'] == 'dry_run',
            )
        else:
            logger.info("Executing entry %s %.4f", direction, size)
            await cex_trade_async(
                exchange,
                ws_client,
                config['symbol'],
                direction,
                size,
                user['telegram_token'],
                user['telegram_chat_id'],
                dry_run=config['execution_mode'] == 'dry_run',
                use_websocket=config.get('use_websocket', False),
                config=config,
            )
            stop_price = current_price * (
                1 - risk_manager.config.stop_loss_pct
                if direction == 'buy'
                else 1 + risk_manager.config.stop_loss_pct
            )
            stop_order = place_stop_order(
                exchange,
                config['symbol'],
                'sell' if direction == 'buy' else 'buy',
                size,
                stop_price,
                user['telegram_token'],
                user['telegram_chat_id'],
                dry_run=config['execution_mode'] == 'dry_run',
            )
            risk_manager.register_stop_order(stop_order)
            open_side = direction
            entry_price = current_price
            position_size = size
            highest_price = entry_price
            logger.info("Trade opened at %.4f", entry_price)

        key = f"{env}_{regime}"
        stats.setdefault(key, {'trades': 0})
        stats[key]['trades'] += 1
        stats_file.write_text(json.dumps(stats))
        logger.info("Updated trade stats %s", stats[key])

        await asyncio.sleep(config['loop_interval_minutes'] * 60)


if __name__ == '__main__':
    asyncio.run(main())

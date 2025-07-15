"""Start the web dashboard and expose REST endpoints for the trading bot.

This module launches the Flask web server, manages the background trading
process and provides REST API routes used by the UI and tests.
"""

from flask import Flask, render_template, redirect, url_for, request, jsonify
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR
import subprocess
import json
import threading
import time
import yaml
from crypto_bot import log_reader
from crypto_bot import ml_signal_model as ml
from . import utils

app = Flask(__name__)

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None
LOG_FILE = LOG_DIR / 'bot.log'
STATS_FILE = LOG_DIR / 'strategy_stats.json'
SCAN_FILE = LOG_DIR / 'asset_scores.json'
MODEL_REPORT = Path('crypto_bot/ml_signal_model/models/model_report.json')
TRADE_FILE = LOG_DIR / 'trades.csv'
ERROR_FILE = LOG_DIR / 'errors.log'
CONFIG_FILE = Path('crypto_bot/config.yaml')
REGIME_FILE = LOG_DIR / 'regime_history.txt'


def watch_bot() -> None:
    """Monitor the trading bot and restart it if the process exits."""
    global bot_proc, bot_start_time
    while True:
        time.sleep(5)
        if bot_proc is not None and bot_proc.poll() is not None:
            bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
            bot_start_time = time.time()



def is_running() -> bool:
    """Return True if the bot process is running."""
    return utils.is_running(bot_proc)


def set_execution_mode(mode: str) -> None:
    """Set execution mode in config file."""
    utils.set_execution_mode(mode, CONFIG_FILE)


def load_execution_mode() -> str:
    """Load execution mode from config file."""
    return utils.load_execution_mode(CONFIG_FILE)


def get_uptime() -> str:
    """Return human readable uptime."""
    return utils.get_uptime(bot_start_time)




@app.route('/')
def index():
    mode = load_execution_mode()
    return render_template(
        'index.html',
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        last_trade=utils.get_last_trade(TRADE_FILE),
        regime=utils.get_current_regime(LOG_FILE),
        last_reason=utils.get_last_decision_reason(LOG_FILE),
    )




@app.route('/start', methods=['POST'])
def start():
    global bot_proc, bot_start_time
    mode = request.form.get('mode', 'dry_run')
    set_execution_mode(mode)
    if not is_running():
        # Launch the asyncio-based trading bot
        bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
        bot_start_time = time.time()
    return redirect(url_for('index'))


@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    mode = (
        request.json.get('mode', 'dry_run') if request.is_json else request.form.get('mode', 'dry_run')
    )
    set_execution_mode(mode)
    status = 'running'
    if not is_running():
        bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
        bot_start_time = time.time()
        status = 'started'
    return jsonify({
        'status': status,
        'running': True,
        'uptime': get_uptime(),
        'mode': mode,
    })


@app.route('/stop')
def stop():
    global bot_proc, bot_start_time
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
    bot_proc = None
    bot_start_time = None
    return redirect(url_for('index'))


@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = 'not_running'
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
        status = 'stopped'
    bot_proc = None
    bot_start_time = None
    return jsonify({
        'status': status,
        'running': False,
        'uptime': get_uptime(),
        'mode': load_execution_mode(),
    })


@app.route('/logs')
def logs_page():
    return render_template('logs.html')


@app.route('/logs_tail')
def logs_tail():
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-200:]
        return '\n'.join(lines)
    return ''


@app.route('/stats')
def stats():
    data = {}
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            data = json.load(f)
    return render_template('stats.html', stats=data)


@app.route('/scans')
def scans():
    data = {}
    if SCAN_FILE.exists():
        with open(SCAN_FILE) as f:
            data = json.load(f)
    return render_template('scans.html', scans=data)


@app.route('/cli', methods=['GET', 'POST'])
def cli():
    """Run CLI commands and display output."""
    output = None
    if request.method == 'POST':
        base = request.form.get('base', 'bot')
        cmd_args = request.form.get('command', '')
        if base == 'backtest':
            cmd = f"python -m crypto_bot.backtest.backtest_runner {cmd_args}"
        elif base == 'custom':
            cmd = cmd_args
        else:
            cmd = f"python -m crypto_bot.main {cmd_args}"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            output = proc.stdout + proc.stderr
        except Exception as exc:  # pragma: no cover - subprocess
            output = str(exc)
    return render_template('cli.html', output=output)
@app.route('/dashboard')
def dashboard():
    summary = log_reader.trade_summary(TRADE_FILE)
    df = log_reader._read_trades(TRADE_FILE)
    perf = utils.compute_performance(df)
    allocation = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get('strategy_allocation', {})
    regimes = []
    if REGIME_FILE.exists():
        regimes = REGIME_FILE.read_text().splitlines()[-20:]
    return render_template(
        'dashboard.html',
        pnl=summary.get('total_pnl', 0.0),
        performance=perf,
        allocation=allocation,
        regimes=regimes,
    )
@app.route('/model')
def model_page():
    report = {}
    if MODEL_REPORT.exists():
        with open(MODEL_REPORT) as f:
            report = json.load(f)
    return render_template('model.html', report=report)


@app.route('/train_model', methods=['POST'])
def train_model_route():
    file = request.files.get('csv')
    if file:
        tmp_path = LOG_DIR / 'upload.csv'
        file.save(tmp_path)
        ml.train_from_csv(tmp_path)
        tmp_path.unlink()
    return redirect(url_for('model_page'))


@app.route('/validate_model', methods=['POST'])
def validate_model_route():
    file = request.files.get('csv')
    tmp_path = None
    if file:
        tmp_path = LOG_DIR / 'validate.csv'
        file.save(tmp_path)
        metrics = ml.validate_from_csv(tmp_path)
        tmp_path.unlink()
    else:
        default_csv = LOG_DIR / 'trades.csv'
        if default_csv.exists():
            metrics = ml.validate_from_csv(default_csv)
        else:
            metrics = ml.validate_from_csv(default_csv)
    if metrics:
        MODEL_REPORT.write_text(json.dumps(metrics))
    return redirect(url_for('model_page'))
@app.route('/trades')
def trades_page():
    return render_template('trades.html')


@app.route('/trades_tail')
def trades_tail():
    trades = ''
    if TRADE_FILE.exists():
        trades = '\n'.join(TRADE_FILE.read_text().splitlines()[-100:])
    errors = ''
    if ERROR_FILE.exists():
        errors = '\n'.join(ERROR_FILE.read_text().splitlines()[-100:])
    return jsonify({'trades': trades, 'errors': errors})


@app.route('/trades_data')
def trades_data():
    """Return full trade history as JSON records."""
    if TRADE_FILE.exists():
        df = log_reader._read_trades(TRADE_FILE)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])


if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    app.run(debug=True)

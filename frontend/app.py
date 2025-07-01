from flask import Flask, render_template, redirect, url_for, request, jsonify
from pathlib import Path
import subprocess
import json
import threading
import time
import psutil
import yaml
from crypto_bot import log_reader
import pandas as pd
from crypto_bot import ml_signal_model as ml

app = Flask(__name__)

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None
LOG_FILE = Path('crypto_bot/logs/bot.log')
STATS_FILE = Path('crypto_bot/logs/strategy_stats.json')
SCAN_FILE = Path('crypto_bot/logs/asset_scores.json')
MODEL_REPORT = Path('crypto_bot/ml_signal_model/models/model_report.json')
TRADE_FILE = Path('crypto_bot/logs/trades.csv')
ERROR_FILE = Path('crypto_bot/logs/errors.log')
CONFIG_FILE = Path('crypto_bot/config.yaml')
TRADES_FILE = Path('crypto_bot/logs/trades.csv')
REGIME_FILE = Path('crypto_bot/logs/regime_history.txt')


def load_execution_mode() -> str:
    """Return execution mode from the YAML config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f).get('execution_mode', 'dry_run')
    return 'dry_run'


def set_execution_mode(mode: str) -> None:
    """Update execution mode in the YAML config."""
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = yaml.safe_load(f) or {}
    config['execution_mode'] = mode
    with open(CONFIG_FILE, 'w') as f:
        yaml.safe_dump(config, f)


def watch_bot():
    """Restart the trading bot if the process exits."""
    global bot_proc, bot_start_time
    while True:
        time.sleep(10)
        if bot_proc and (bot_proc.poll() is not None or not psutil.pid_exists(bot_proc.pid)):
            bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
            bot_start_time = time.time()


def compute_performance(df: pd.DataFrame) -> dict:
    """Return PnL per symbol from the trades dataframe."""
    perf: dict[str, float] = {}
    open_pos: dict[str, list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        symbol = row.get("symbol")
        side = row.get("side")
        price = float(row.get("price", 0))
        amount = float(row.get("amount", 0))
        if side == "buy":
            open_pos.setdefault(symbol, []).append((price, amount))
        elif side == "sell":
            lst = open_pos.setdefault(symbol, [])
            while amount > 0 and lst:
                entry_price, qty = lst.pop(0)
                traded = min(qty, amount)
                perf[symbol] = perf.get(symbol, 0.0) + (price - entry_price) * traded
                if qty > traded:
                    lst.insert(0, (entry_price, qty - traded))
                amount -= traded
    return perf


def is_running() -> bool:
    return bot_proc is not None and bot_proc.poll() is None


def get_uptime() -> str:
    """Return human-readable uptime."""
    if bot_start_time is None:
        return "-"
    delta = int(time.time() - bot_start_time)
    hrs, rem = divmod(delta, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def get_last_trade() -> str:
    """Return last trade from trades CSV."""
    if not TRADE_FILE.exists():
        return "N/A"
    import csv

    with open(TRADE_FILE) as f:
        rows = list(csv.reader(f))
    if not rows:
        return "N/A"
    row = rows[-1]
    if len(row) >= 4:
        sym, side, amt, price = row[:4]
        return f"{side} {amt} {sym} @ {price}"
    return "N/A"


def get_current_regime() -> str:
    """Return most recent regime classification from bot log."""
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()
        for line in reversed(lines):
            if "Market regime classified as" in line:
                return line.rsplit("Market regime classified as", 1)[1].strip()
    return "N/A"


@app.route('/')
def index():
    mode = load_execution_mode()
    return render_template(
        'index.html',
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        last_trade=get_last_trade(),
        regime=get_current_regime(),
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


@app.route('/stop')
def stop():
    global bot_proc, bot_start_time
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
    bot_proc = None
    bot_start_time = None
    return redirect(url_for('index'))


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
    summary = log_reader.trade_summary(TRADES_FILE)
    df = log_reader._read_trades(TRADES_FILE)
    perf = compute_performance(df)
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
        tmp_path = Path('crypto_bot/logs/upload.csv')
        file.save(tmp_path)
        ml.train_from_csv(tmp_path)
        tmp_path.unlink()
    return redirect(url_for('model_page'))


@app.route('/validate_model', methods=['POST'])
def validate_model_route():
    file = request.files.get('csv')
    tmp_path = None
    if file:
        tmp_path = Path('crypto_bot/logs/validate.csv')
        file.save(tmp_path)
        metrics = ml.validate_from_csv(tmp_path)
        tmp_path.unlink()
    else:
        default_csv = Path('crypto_bot/logs/trades.csv')
        if default_csv.exists():
            metrics = ml.validate_from_csv(default_csv)
        else:
            metrics = {}
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


if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    app.run(debug=True)

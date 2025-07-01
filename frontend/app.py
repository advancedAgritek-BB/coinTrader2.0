from flask import Flask, render_template, redirect, url_for, request
from pathlib import Path
import subprocess
import json
import threading
import time
import psutil
import yaml

app = Flask(__name__)

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None
LOG_FILE = Path('crypto_bot/logs/bot.log')
STATS_FILE = Path('crypto_bot/logs/strategy_stats.json')
SCAN_FILE = Path('crypto_bot/logs/asset_scores.json')
TRADE_FILE = Path('crypto_bot/logs/trades.csv')
CONFIG_FILE = Path('crypto_bot/config.yaml')


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


if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    app.run(debug=True)

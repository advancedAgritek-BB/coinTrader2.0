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
watch_thread = None
LOG_FILE = Path('crypto_bot/logs/bot.log')
STATS_FILE = Path('crypto_bot/logs/strategy_stats.json')
SCAN_FILE = Path('crypto_bot/logs/asset_scores.json')
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
    global bot_proc
    while True:
        time.sleep(10)
        if bot_proc and (bot_proc.poll() is not None or not psutil.pid_exists(bot_proc.pid)):
            bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])




def is_running() -> bool:
    return bot_proc is not None and bot_proc.poll() is None


@app.route('/')
def index():
    mode = load_execution_mode()
    return render_template('index.html', running=is_running(), mode=mode)


@app.route('/start', methods=['POST'])
def start():
    global bot_proc
    mode = request.form.get('mode', 'dry_run')
    set_execution_mode(mode)
    if not is_running():
        # Launch the asyncio-based trading bot
        bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
    return redirect(url_for('index'))


@app.route('/stop')
def stop():
    global bot_proc
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
    bot_proc = None
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


if __name__ == '__main__':
    watch_thread = threading.Thread(target=watch_bot, daemon=True)
    watch_thread.start()
    app.run(debug=True)

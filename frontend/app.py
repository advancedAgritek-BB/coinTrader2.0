from flask import Flask, render_template, redirect, url_for
from pathlib import Path
import subprocess
import json
import threading
import time
import psutil

app = Flask(__name__)

# Handle the async trading bot process
bot_proc = None
LOG_FILE = Path('crypto_bot/logs/bot.log')
STATS_FILE = Path('crypto_bot/logs/strategy_stats.json')


def watch_bot():
    """Restart the trading bot if the process exits."""
    global bot_proc
    while True:
        time.sleep(10)
        if bot_proc and (bot_proc.poll() is not None or not psutil.pid_exists(bot_proc.pid)):
            bot_proc = subprocess.Popen(['python', '-m', 'crypto_bot.main'])


watch_thread = threading.Thread(target=watch_bot, daemon=True)
watch_thread.start()


def is_running() -> bool:
    return bot_proc is not None and bot_proc.poll() is None


@app.route('/')
def index():
    return render_template('index.html', running=is_running())


@app.route('/start')
def start():
    global bot_proc
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


if __name__ == '__main__':
    app.run(debug=True)

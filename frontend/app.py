from flask import Flask, render_template, redirect, url_for
from pathlib import Path
import subprocess
import json

app = Flask(__name__)

bot_process = None
LOG_FILE = Path('crypto_bot/logs/bot.log')
STATS_FILE = Path('crypto_bot/logs/strategy_stats.json')


def is_running() -> bool:
    return bot_process is not None and bot_process.poll() is None


@app.route('/')
def index():
    return render_template('index.html', running=is_running())


@app.route('/start')
def start():
    global bot_process
    if not is_running():
        bot_process = subprocess.Popen(['python', '-m', 'crypto_bot.main'])
    return redirect(url_for('index'))


@app.route('/stop')
def stop():
    global bot_process
    if is_running():
        bot_process.terminate()
        bot_process.wait()
    bot_process = None
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

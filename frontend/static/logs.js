const LOG_POLL_INTERVAL = 3000; // milliseconds

async function fetchLogs() {
  try {
    const resp = await fetch('/logs_tail');
    const text = await resp.text();
    const logArea = document.getElementById('log-area');
    if (logArea) {
      logArea.textContent = text;
      if (window.hljs) {
        hljs.highlightElement(logArea);
      }
    }
  } catch (err) {
    console.error('Failed to fetch logs', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchLogs();
  setInterval(fetchLogs, LOG_POLL_INTERVAL);
});

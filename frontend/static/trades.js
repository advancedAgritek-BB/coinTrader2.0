const TRADE_POLL_INTERVAL = 5000; // milliseconds

function buildRow(data) {
  const row = document.createElement('tr');
  if (data.side === 'buy') {
    row.classList.add('table-success');
  } else if (data.side === 'sell') {
    row.classList.add('table-danger');
  }
  row.innerHTML = `<td>${data.symbol || ''}</td><td>${data.side || ''}</td><td>${data.amount || ''}</td><td>${data.price || ''}</td><td>${data.timestamp || ''}</td>`;
  return row;
}

async function fetchTrades() {
  try {
    const resp = await fetch('/trades_data');
    const trades = await resp.json();
    const tbody = document.getElementById('trades-body');
    if (tbody) {
      tbody.innerHTML = '';
      trades.slice(-100).forEach(t => tbody.appendChild(buildRow(t)));
    }
    const logResp = await fetch('/trades_tail');
    const logs = await logResp.json();
    const errorArea = document.getElementById('errors-area');
    if (errorArea) {
      errorArea.textContent = logs.errors;
      if (window.hljs) {
        hljs.highlightElement(errorArea);
      }
    }
  } catch (err) {
    console.error('Failed to fetch trades', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchTrades();
  setInterval(fetchTrades, TRADE_POLL_INTERVAL);
});

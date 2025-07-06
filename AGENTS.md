# 📄 agents.md – Kraken API Reference for Codex Agent Use

## 🧭 Kraken Spot API Overview

Kraken provides two primary APIs for spot trading:

1. **REST API** — for polling market data and executing trades  
2. **WebSocket v2 API** — for real-time updates and order lifecycle events

---

## ✅ REST API Summary

### 🔗 Base URL
https://api.kraken.com/0/

### 🔓 Public Endpoints

| Endpoint                      | Description                       |
|-------------------------------|-----------------------------------|
| /public/SystemStatus          | Check exchange status             |
| /public/Time                  | Server time                       |
| /public/Ticker?pair=XBTUSD    | Market ticker                     |
| /public/OHLC?pair=XBTUSD      | Historical candles                |
| /public/Depth?pair=XBTUSD     | Order book snapshot               |
| /public/AssetPairs            | Supported symbols                 |

#### 🧪 Example
GET /0/public/Ticker?pair=XBTUSD

```json
{
  "error": [],
  "result": {
    "XXBTZUSD": {
      "a": ["40357.20000", "1", "1.000"],
      "b": ["40357.10000", "1", "1.000"],
      "c": ["40357.10000", "0.001"]
    }
  }
}
```

### 🔐 Private Endpoints (require API Key)

| Endpoint            | Description              |
|---------------------|--------------------------|
| /private/Balance    | Get account balances     |
| /private/TradeBalance | Get current margin balance |
| /private/OpenOrders | List open orders         |
| /private/ClosedOrders | Past orders            |
| /private/AddOrder   | Place order              |
| /private/CancelOrder| Cancel order             |

#### 🧪 Add Order Example

POST /0/private/AddOrder

Form data:
```
pair=XBTUSD
type=buy
ordertype=limit
price=39000
volume=0.01
```

Headers:
```
API-Key: YOUR_API_KEY
API-Sign: SHA512-HMAC of message using secret
```

## 📡 WebSocket v2 API Summary

**Public WS Endpoint**
wss://ws.kraken.com/v2

**Private WS Endpoint**
wss://ws-auth.kraken.com/v2

The `KrakenWSClient` automatically processes `status` and `heartbeat` channels,
so explicit subscriptions for those are unnecessary.

### Subscribe to Market Data
Ticker

```json
{
  "method": "subscribe",
  "params": {
    "channel": "ticker",
    "symbol": ["XBT/USD"]
  }
}
```
Order Book

```json
{
  "method": "subscribe",
  "params": {
    "channel": "book",
    "symbol": ["ETH/USD"],
    "depth": 10,
    "snapshot": true
  }
}
```
Place Order via WebSocket
Call the REST API /private/GetWebSocketsToken
Use that token in the request payload

```json
{
  "method": "add_order",
  "params": {
    "symbol": "XBT/USD",
    "side": "buy",
    "order_type": "limit",
    "price": "39000.0",
    "order_qty": "0.01"
  },
  "token": "YOUR_WS_TOKEN"
}
```
Balance + Trade Events
Subscribe to private channels after authentication:

```json
{
  "method": "subscribe",
  "params": {
    "channel": "executions",
    "token": "YOUR_WS_TOKEN"
  }
}
```

### 🔐 Authentication Flow

1. Generate nonce (increasing integer)
2. Construct message: path + SHA256(nonce + POST data)
3. Sign with HMAC-SHA512(secret)
4. Add headers:
API-Key
API-Sign

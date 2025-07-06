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

### \U1F4DA Book Channel

The order book is streamed via the public WebSocket endpoint:

```
wss://ws.kraken.com/v2
```

Subscribe with:

```json
{
  "method": "subscribe",
  "params": {
    "channel": "book",
    "symbol": ["XBT/USD"],
    "depth": 10,
    "snapshot": true
  }
}
```

Snapshot messages include `a` and `b` arrays for ask and bid levels,
`checksum`, `symbol`, `channel`, `timestamp` and `sequence` numbers.
Subsequent updates set `"snapshot": false` and only send changed levels
in `a` or `b`.

### 🔒 Level 3 Orders Channel

Authenticated clients can subscribe to the full depth book using the
`level3` channel. Only one depth level per symbol is allowed (10, 100 or
1000). A session token obtained via the REST `GetWebSocketsToken` endpoint
is required.

Example subscribe request:

```json
{
  "method": "subscribe",
  "params": {
    "channel": "level3",
    "symbol": ["ALGO/USD", "MATIC/USD"],
    "snapshot": true,
    "token": "YOUR_WS_TOKEN"
  }
}
```

Each symbol receives a separate acknowledgement:

```json
{
  "method": "subscribe",
  "result": {
    "channel": "level3",
    "snapshot": true,
    "symbol": "ALGO/USD"
  },
  "success": true,
  "time_in": "2023-10-06T18:20:56.506266Z",
  "time_out": "2023-10-06T18:20:56.521803Z"
}
```

Snapshot messages provide all visible orders:

```json
{
  "channel": "level3",
  "type": "snapshot",
  "data": [
    {
      "symbol": "MATIC/USD",
      "checksum": 281817320,
      "bids": [
        {
          "order_id": "O6ZQNQ-BXL4E-5WGINO",
          "limit_price": 0.5629,
          "order_qty": 111.56125344,
          "timestamp": "2023-10-06T17:35:00.279389650Z"
        }
      ],
      "asks": [
        {
          "order_id": "OLLSXO-HDMT3-BUOKEI",
          "limit_price": 0.563,
          "order_qty": 4422.9978357,
          "timestamp": "2023-10-06T18:18:20.734897896Z"
        }
      ]
    }
  ]
}
```

Updates share the same structure but use `"type": "update"` and only
include changed orders. The `checksum` field validates the top levels of
the book.

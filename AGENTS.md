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
Cancel Order via WebSocket
Authenticated endpoint:

```
wss://ws-auth.kraken.com/v2
```

Use `cancel_order` to close one or more open orders. Provide Kraken
`order_id` or your own `cl_ord_id` references. Each cancelled order will
also appear on the `executions` stream.

```json
{
    "method": "cancel_order",
    "params": {
        "order_id": ["OM5CRX-N2HAL-GFGWE9", "OLUMT4-UTEGU-ZYM7E9"],
        "token": "YOUR_WS_TOKEN"
    },
    "req_id": 123456789
}
```

```json
{
  "method": "cancel_order",
  "req_id": 123456789,
  "result": {"order_id": "OLUMT4-UTEGU-ZYM7E9"},
  "success": true,
  "time_in": "2023-09-21T14:36:57.428972Z",
  "time_out": "2023-09-21T14:36:57.437952Z"
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
### \U1F4C8 OHLC Channel

Kraken streams candlestick data through the public WebSocket endpoint:

```
wss://ws.kraken.com/v2
```

Subscribe with the following fields:

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
    "channel": "ohlc",
    "symbol": ["XBT/USD"],
    "interval": 1,
    "snapshot": true
  },
  "req_id": 42
}
```

Example ACK response:

```json
{
  "channel": "ohlc",
  "event": "subscribe",
  "req_id": 42,
  "status": "ok"
}
```

Snapshot and update messages contain the candle fields:
`open`, `high`, `low`, `close`, `vwap`, `volume`, `trades`,
`interval_begin`, `interval`, and `snapshot`.

Snapshot example:

```json
{
  "channel": "ohlc",
  "symbol": "XBT/USD",
  "interval": 1,
  "interval_begin": 1682305800,
  "open": "28885.4",
  "high": "28888.0",
  "low": "28880.1",
  "close": "28882.3",
  "vwap": "28883.1",
  "volume": "12.3456",
  "trades": 25,
  "snapshot": true
}
```

Update example:

```json
{
  "channel": "ohlc",
  "symbol": "XBT/USD",
  "interval": 1,
  "interval_begin": 1682305860,
  "open": "28882.3",
  "high": "28890.5",
  "low": "28881.0",
  "close": "28889.9",
  "vwap": "28888.0",
  "volume": "3.2100",
  "trades": 7,
  "snapshot": false
}
```

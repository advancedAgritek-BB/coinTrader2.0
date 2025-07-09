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

### \U1F4E6 Add Order via WebSocket

**Endpoint**: `wss://ws-auth.kraken.com/v2`

Send a JSON payload with `"method": "add_order"` and the following parameters.

#### Request Schema

| Field | Type | Description |
|-------|------|-------------|
| `order_type` | string | Order execution model, e.g. `limit`, `market`, `iceberg`, `stop-loss`, etc. |
| `side` | string | `buy` or `sell`. |
| `order_qty` | float | Quantity in base asset. |
| `symbol` | string | Trading pair like `"BTC/USD"`. |
| `limit_price` | float | Optional limit price for supported order types. |
| `limit_price_type` | string | Units for limit price (`static`, `pct`, `quote`). |
| `triggers` | object | Trigger parameters for stop and trailing orders. |
| `time_in_force` | string | `gtc`, `gtd`, or `ioc`. |
| `margin` | boolean | Enable margin funding. |
| `post_only` | boolean | Only post if it adds liquidity. |
| `reduce_only` | boolean | Reduce existing position only. |
| `effective_time` | string | RFC3339 scheduled start time. |
| `expire_time` | string | RFC3339 expiration time (GTD only). |
| `deadline` | string | Max lifetime before matching. |
| `cl_ord_id` | string | Optional client supplied order id. |
| `order_userref` | integer | Optional numeric order reference. |
| `conditional` | object | For OTO orders, defines the secondary close order. |
| `display_qty` | float | Iceberg display quantity. |
| `fee_preference` | string | `base` or `quote` fee currency. |
| `no_mpp` | boolean | Disable Market Price Protection for market orders. |
| `stp_type` | string | Self trade prevention mode. |
| `cash_order_qty` | float | Quote currency volume for buy market orders. |
| `validate` | boolean | Validate only without trading. |
| `sender_sub_id` | string | Sub-account identifier. |
| `token` | string | WebSocket authentication token. |
| `req_id` | integer | Optional request identifier echoed back. |

Example limit order:

```json
{
    "method": "add_order",
    "params": {
        "order_type": "limit",
        "side": "buy",
        "limit_price": 26500.4,
        "order_userref": 100054,
        "order_qty": 1.2,
        "symbol": "BTC/USD",
        "token": "G38a1tGFzqGiUCmnegBcm8d4nfP3tytiNQz6tkCBYXY"
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
Example stop-loss:

```json
{
    "method": "add_order",
    "params": {
        "order_type": "stop-loss",
        "side": "sell",
        "order_qty": 100,
        "symbol": "MATIC/USD",
        "triggers": {
            "reference": "last",
            "price": -2.0,
            "price_type": "pct"
        },
        "token": "G38a1tGFzqGiUCmnegBcm8d4nfP3tytiNQz6tkCBYXY"
    }
}
```

One-Triggers-Other example:

```json
{
    "method": "add_order",
    "params": {
        "order_type": "limit",
        "side": "buy",
        "order_qty": 1.2,
        "symbol": "BTC/USD",
        "limit_price": 28440,
        "conditional": {
            "order_type": "stop-loss-limit",
            "trigger_price": 28410,
            "limit_price": 28400
        },
        "token": "G38a1tGFzqGiUCmnegBcm8d4nfP3tytiNQz6tkCBYXY"
    }
}
```

#### Response

Successful responses echo the `req_id` and return `order_id` along with any
optional references:

```json
{
    "method": "add_order",
    "req_id": 123456789,
    "result": {
        "order_id": "AA5JGQ-SBMRC-SCJ7J7",
        "order_userref": 100054
    },
    "success": true,
    "time_in": "2023-09-21T14:15:07.197274Z",
    "time_out": "2023-09-21T14:15:07.205301Z"
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

## 🔄 Amending Orders via WebSocket

Kraken allows existing orders to be modified in place using the authenticated `amend_order` method. Queue priority is kept where possible and the order identifiers remain the same. If the new quantity is below the already filled amount, the remainder is cancelled.

### Request Fields
- **order_id** or **cl_ord_id** – identify the order to amend
- **order_qty** – new base asset quantity
- **display_qty** – visible portion for iceberg orders (\>= 1/15 of remaining)
- **limit_price** and **limit_price_type** – updated limit price information
- **post_only** – reject the amend if it would take liquidity
- **trigger_price** and **trigger_price_type** – for stop or trailing orders
- **deadline** – RFC3339 timestamp, max 60s in the future
- **token** – WebSocket auth token
- **req_id** – optional client request ID

### Basic Example
```json
{
  "method": "amend_order",
  "params": {
    "cl_ord_id": "2c6be801-1f53-4f79-a0bb-4ea1c95dfae9",
    "limit_price": 490795,
    "order_qty": 1.2,
    "token": "PM5Qm0MDrS54l657aQAtb7AhrwN30e2LBg1nUYOd6vU"
  }
}
```

### Advanced Example
```json
{
  "method": "amend_order",
  "params": {
    "order_id": "OAIYAU-LGI3M-PFM5VW",
    "limit_price": 61031.3,
    "deadline": "2024-07-21T09:53:59.050Z",
    "post_only": true,
    "token": "DGB00LiKlPlLI/amQaSKUUr8niqXDb+1zwvtjp34nzk"
  }
}
```

### Response
Successful requests return a unique `amend_id` and echo the identifiers used. The response also includes timestamps `time_in` and `time_out`.
```json
{
  "method": "amend_order",
  "result": {
    "amend_id": "TTW6PD-RC36L-ZZSWNU",
    "cl_ord_id": "2c6be801-1f53-4f79-a0bb-4ea1c95dfae9"
  },
  "success": true,
  "time_in": "2024-07-26T13:39:04.922699Z",
  "time_out": "2024-07-26T13:39:04.924912Z"
}
```

### Notes
- Only quantity, display quantity, limit price and trigger price can be changed. Other attributes require cancelling and re-adding the order.
- Setting `post_only` ensures the order remains passive after the amend.
- Iceberg display size must be at least 1/15 of the remaining size.
- Orders with conditional close terms cannot be amended.


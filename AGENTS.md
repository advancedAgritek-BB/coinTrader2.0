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

### ✏️ Edit Order via WebSocket

**Endpoint**: `wss://ws-auth.kraken.com/v2`

Send a JSON payload with `"method": "edit_order"` to modify an existing open order. The original order is canceled and replaced with a new one that has a new `order_id`.

**Caveats**

* Triggered stop-loss or take-profit orders are not supported.
* Orders with conditional close terms attached are not supported.
* Requests where the executed volume exceeds the newly supplied volume are rejected.
* `cl_ord_id` is not supported.
* Existing executions remain associated with the original order.
* Queue position is not maintained.

#### Request Schema

| Field | Type | Description |
|-------|------|-------------|
| `method` | string | Must be `edit_order`. |
| `order_id` | string | Kraken order identifier to amend. |
| `symbol` | string | Original trading pair. |
| `token` | string | WebSocket authentication token. |
| `order_qty` | float | New order quantity. |
| `limit_price` | float | Optional limit price. |
| `display_qty` | float | Iceberg display quantity. |
| `fee_preference` | string | `base` or `quote` fee currency. |
| `no_mpp` | boolean | Disable Market Price Protection for market orders. |
| `post_only` | boolean | Cancel if it would take liquidity. |
| `reduce_only` | boolean | Reduce existing position only. |
| `triggers` | object | Trigger price parameters. |
| `deadline` | string | RFC3339 timestamp after which the order will not match. |
| `order_userref` | integer | User reference for the amended order. |
| `validate` | boolean | If `true`, validate only. |
| `req_id` | integer | Client request identifier echoed back. |

Example request:

```json
{
  "method": "edit_order",
  "params": {
    "order_id": "ORDERX-IDXXX-XXXXX1",
    "order_qty": 0.2123456789,
    "symbol": "BTC/USD",
    "token": "TxxxxxxxxxOxxxxxxxxxxKxxxxxxxExxxxxxxxN"
  },
  "req_id": 1234567890
}
```

#### Response

Successful responses return the new `order_id` along with the original one:

```json
{
  "method": "edit_order",
  "req_id": 1234567890,
  "result": {
    "order_id": "ORDERX-IDXXX-XXXXX2",
    "original_order_id": "ORDERX-IDXXX-XXXXX1"
  },
  "success": true,
  "time_in": "2022-07-15T12:56:09.876488Z",
  "time_out": "2022-07-15T12:56:09.923422Z"
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


## 🚀 Helius API Quickstart

The Helius Dashboard provides a free tier with 100,000 DAS API calls per month. After creating an account, navigate to **API Keys** and copy your key. This key grants access to RPC nodes, the DAS API, and enhanced transaction data.

### 📟 Fetching NFTs by Owner

Helius' `getAssetsByOwner` method returns compressed and standard NFTs with a single request. The call structure is:

```json
POST https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "getAssetsByOwner",
  "params": {
    "ownerAddress": "WALLET_ADDRESS",
    "page": 1,
    "limit": 10,
    "displayOptions": {
      "showFungible": false,
      "showNativeBalance": false
    }
  }
}
```

The response includes the NFT ID, metadata (name, symbol, description), content files (such as image URLs), and ownership details.

### 🛠️ Example Portfolio Viewer

Below is a minimal Node.js script that prints an NFT portfolio summary. Install `node-fetch` and replace `YOUR_API_KEY` with the key from your dashboard.

```bash
mkdir solana-nft-viewer
cd solana-nft-viewer
npm init -y
npm install node-fetch
```

Create `nft-portfolio.js`:

```javascript
const fetch = require('node-fetch');

class NFTPortfolioViewer {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseUrl = 'https://mainnet.helius-rpc.com';
  }

  async fetchNFTsByOwner(ownerAddress, limit = 10) {
    const res = await fetch(`${this.baseUrl}/?api-key=${this.apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '1',
        method: 'getAssetsByOwner',
        params: { ownerAddress, page: 1, limit,
          displayOptions: { showFungible: false, showNativeBalance: false } }
      })
    });
    return res.json();
  }
}

(async () => {
  const viewer = new NFTPortfolioViewer('YOUR_API_KEY');
  const wallet = '86xCnPeV69n6t3DnyGvkKobf9FdN2H9oiVDdaMpo2MMY';
  const portfolio = await viewer.fetchNFTsByOwner(wallet);
  console.log(JSON.stringify(portfolio, null, 2));
})();
```

Run the viewer with:

```bash
node nft-portfolio.js
```

This prints a summary of NFTs in the wallet, including collection info and compression status.

### Fetching New Pools

Helius replaced the old `getPools` RPC method with `dex.getNewPools`. Use the standard RPC endpoint and include a protocols list:

```json
POST https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "dex.getNewPools",
  "params": {"protocols": ["raydium"], "limit": 50}
}
```


## \ud83d\udce1 Helius Data Streaming \u2013 Quickstart Guide

Helius offers multiple real-time data streaming options for the Solana blockchain, each tailored for different use cases depending on your performance, filtering, and integration needs. Below is a complete breakdown of the streaming methods available:

### \u2705 Overview of Streaming Methods

| Method | Use Case | Latency | Plan | Best For |
|-------|----------|---------|------|---------|
| **Standard WebSockets** | General-purpose streaming with native support | Good | Free | Simple apps, existing Solana WebSocket integrations |
| **Enhanced WebSockets** | High-performance filters and fast response | Fast | Business | dApps, wallets, high-frequency event processing |
| **LaserStream (gRPC)** | Ultra-low latency with historical replay | Fastest | Professional | HFT systems, block explorers, serious infra |
| **Webhooks** | Server-side integrations via HTTP callbacks | Variable | Free | Notifications, backend systems, Discord bots, etc. |

### \ud83d\udd0c Standard WebSockets

Endpoint: `wss://mainnet.helius-rpc.com`

Helius recently migrated from `rpc.helius.xyz` to `mainnet.helius-rpc.com`. Make sure your configuration uses the new domain.

Features:

- Works with any Solana-compatible WebSocket client.
- Direct drop-in replacement for `solana/web3.js` `onAccountChange`.
- Use your Helius API key in the endpoint.

Example:

```javascript
import { Connection, clusterApiUrl, PublicKey } from "@solana/web3.js";

const connection = new Connection("wss://mainnet.helius-rpc.com/?api-key=<YOUR_API_KEY>");

connection.onAccountChange(
  new PublicKey("TARGET_PUBLIC_KEY"),
  (updatedAccountInfo) => {
    console.log("Account Updated", updatedAccountInfo);
  }
);
```

### \u26a1 Enhanced WebSockets

Endpoint: `wss://stream.helius.xyz/v0/transactions`

Features:

- Real-time transaction streaming.
- Filters by accounts, transaction types, and more.
- Supports dynamic subscriptions.

Subscription Schema:

```json
{
  "apiKey": "<YOUR_API_KEY>",
  "type": "subscribe",
  "topic": "transactions",
  "id": "unique-subscription-id",
  "accounts": ["<ACCOUNT_1>", "<ACCOUNT_2>"]
}
```

Example Response:

```json
{
  "type": "transaction",
  "data": {
    "signature": "...",
    "slot": 123456,
    "timestamp": 1712345678
  }
}
```

### \ud83d\ude80 LaserStream (gRPC)

Endpoint: `https://grpc.helius.xyz`

Features:

- Built using gRPC.
- Supports historical data replay.
- Multi-node failover.
- Filtered transaction streams.
- Ultra-low latency.
- Advanced error handling.

Requirements:

- gRPC client (e.g. Go, Rust, Node.js)
- Protobufs available in the Helius GitHub.

Example Use Case:

```pseudo
client.SubscribeTransactions(account: "TARGET_PUBKEY", filters: [...])
```

Ideal for:

- Market makers
- Indexers
- Block explorers
- Bots requiring sub-second latency

### \ud83d\udcec Webhooks

Endpoint: `https://api.helius.xyz/v0/webhooks`

Types:

- Wallet Activity (deposits, transfers)
- NFT Sales
- Token Events
- Custom Programs

Features:

- Push-based (no persistent connection required).
- Works with any backend.
- Reliable retries.
- Filter on accounts, program IDs, etc.

Webhook Payload Example:

```json
{
  "eventType": "TRANSFER",
  "signature": "...",
  "source": "SourceWallet",
  "destination": "DestinationWallet",
  "amount": 1000
}
```

Setup:

1. Create a webhook via the Helius Dashboard or API.
2. Configure target URL, filters, and secret key.
3. Handle incoming POST requests.

### \ud83e\udde0 Choosing the Right Stream

| App Type | Recommended Method |
|----------|-------------------|
| Frontend dApp | Standard or Enhanced WS |
| HFT Bot | LaserStream |
| Backend App | Webhooks or LaserStream |
| Indexing System | LaserStream |
| Discord Alerts | Webhooks |

### \ud83d\udd27 Getting Started

1. Sign up at: <https://www.helius.xyz>
2. Generate an API Key in your dashboard.
3. Pick your stream method based on your app needs.
4. Integrate using the examples above.

### \ud83d\udcda Docs & Resources

- [Helius Docs Homepage](https://docs.helius.xyz)
- [Data Streaming Overview](https://docs.helius.xyz/data-streaming)
- [gRPC Protocol Buffers](https://github.com/helius-labs/helius-sdk-protobufs)
- [Enhanced WebSocket Blog](https://www.helius.dev/blog/enhanced-websockets)


## ⚓ Helius Standard WebSocket Methods
### \u2693 WebSocket Connection

Endpoint:
- **Mainnet:** `wss://mainnet.helius-rpc.com/?api-key=<API_KEY>`
- **Devnet:** `wss://devnet.helius-rpc.com/?api-key=<API_KEY>`

Connections idle for 10 minutes. Send periodic pings (e.g., every minute) to keep the socket alive.

### accountSubscribe / accountUnsubscribe
**Params**
- `pubkey` (string) – account public key (base-58)
- `config` (object)
  - `encoding`: `"base58"`, `"base64"`, `"base64+zstd"`, or `"jsonParsed"`
  - `commitment`: `"processed"`, `"confirmed"`, or `"finalized"`

**Result**
- integer `subscriptionId`

**Notification**
```json
{
  "jsonrpc":"2.0",
  "method":"accountNotification",
  "params":{
    "subscription":<id>,
    "result":{
      "context":{"slot":<number>},
      "value":{
        "data":[<data>,<encoding>],
        "executable":<bool>,
        "lamports":<number>,
        "owner":<pubkey>,
        "rentEpoch":<number>
      }
    }
  }
}
```

Unsubscribe via `accountUnsubscribe([subscriptionId])` → returns `{ "result": true }` on success.

### programSubscribe / programUnsubscribe
**Params**
- `programId` (string) – program public key (base-58)
- `config` (object)
  - `encoding`
  - `commitment`
  - optional `filters` (e.g., `{ "dataSize": 80 }`)

**Result**
- integer `subscriptionId`

**Notification**
Same structure as `getProgramAccounts`, triggered when account lamports or data change.

Unsubscribe via `programUnsubscribe(subscriptionId)` → returns `{ "result": true }`.

### logsSubscribe / logsUnsubscribe
**Params**
- `filter`: `"all"`, `"allWithVotes"`, or `{ "mentions": [<pubkey>] }`
- `config`
  - `commitment`

**Result**
- integer `subscriptionId`

**Notification**
```json
{"value":{"signature":<string>,"err":<object|null>,"logs":[<string>,...]}}
```

Unsubscribe with `logsUnsubscribe(subscriptionId)`.

### blockSubscribe / blockUnsubscribe
No params. Returns a `subscriptionId`. Notification is empty for new block confirmations. Unsubscribe with `blockUnsubscribe(subscriptionId)`.

### slotSubscribe / slotUnsubscribe
No params. Returns a `subscriptionId`.
Notification example:
```json
{"parent":<u64>,"root":<u64>,"slot":<u64>}
```
Unsubscribe with `slotUnsubscribe(subscriptionId)`.

### slotsUpdatesSubscribe / slotsUpdatesUnsubscribe
No params. Provides detailed slot lifecycle events. Unsubscribe with `slotsUpdatesUnsubscribe(subscriptionId)`.

### rootSubscribe / rootUnsubscribe
No params. Returns a `subscriptionId`. Notification indicates a new finalized root slot. Unsubscribe with `rootUnsubscribe(subscriptionId)`.

### signatureSubscribe / signatureUnsubscribe
**Params**
1. `signature` (string)
2. `config` with `commitment`

Result is a `subscriptionId`. Notification delivers confirmation status.
Unsubscribe with `signatureUnsubscribe(subscriptionId)`.

### voteSubscribe / voteUnsubscribe
No params. Returns a `subscriptionId`. Notification reports new vote activity. Unsubscribe with `voteUnsubscribe(subscriptionId)`.

### Unsubscribe Pattern
All unsub methods follow:
```json
{"jsonrpc":"2.0","id":1,"method":"unsubscribe","params":[<subscriptionId>]}
```
If successful, the response is `{ "result": true }`.

### \u2728 Enhanced WebSockets (Beta)
Helius also offers **Enhanced WebSockets** via `atlas-mainnet.helius-rpc.com` and `atlas-devnet.helius-rpc.com` for Business/Professional plans. These use the same subscription semantics but with Geyser-enhanced delivery.

### \u2705 Summary Table
| Method | Params | Event Data |
|-------|-------|------------|
| accountSubscribe | pubkey, config | Account data changes |
| programSubscribe | programId, filters, config | Owned-account changes |
| logsSubscribe | filter, config | Transaction log messages |
| signatureSubscribe | signature, config | Transaction status/confirmation |
| blockSubscribe | none | New block notification |
| slotSubscribe | none | Slot processed notification |
| slotsUpdatesSubscribe | none | Detailed slot lifecycle events |
| rootSubscribe | none | New finalized root notification |
| voteSubscribe | none | Vote events observed in the network |


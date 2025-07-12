import pandas as pd
from cachetools import LRUCache
from crypto_bot import main


def test_update_df_cache_global_lru():
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    global_cache = LRUCache(maxsize=2)
    df = pd.DataFrame({"close": [1]})
    main.update_df_cache(cache, "1h", "BTC/USD", df, max_size=1, global_cache=global_cache)
    assert ("1h", "BTC/USD") in global_cache
    main.update_df_cache(cache, "1h", "ETH/USD", df, max_size=1, global_cache=global_cache)
    assert "BTC/USD" not in cache["1h"]
    assert ("1h", "BTC/USD") not in global_cache
    assert ("1h", "ETH/USD") in global_cache

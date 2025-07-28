import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fetch_snapshot(mint: str, bucket: str = "snapshots") -> Optional[dict]:
    """Return JSON snapshot for ``mint`` from Supabase storage.

    Parameters
    ----------
    mint:
        Token mint address used as the file name without extension.
    bucket:
        Storage bucket containing the snapshot JSON file. Defaults to
        ``"snapshots"``.

    Returns
    -------
    dict | None
        Parsed snapshot data, or ``None`` when the download fails or
        credentials/SDK are unavailable.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.error("Missing Supabase credentials")
        return None

    try:
        from supabase import create_client  # type: ignore
    except Exception as exc:
        logger.error("Supabase client unavailable: %s", exc)
        return None

    try:
        client = create_client(url, key)
        data = client.storage.from_(bucket).download(f"{mint}.json")
        return json.loads(data)
    except Exception as exc:
        logger.error("Failed to fetch snapshot %s: %s", mint, exc)
        return None

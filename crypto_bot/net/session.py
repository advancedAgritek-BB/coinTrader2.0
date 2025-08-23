import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(pool: int = 25) -> requests.Session:
    """Create a :class:`requests.Session` with an expanded connection pool.

    Parameters
    ----------
    pool : int, optional
        Connection pool size for both ``pool_connections`` and ``pool_maxsize``.
        Defaults to ``25`` to better match Kraken's HTTP pool.
    """

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool,
        pool_maxsize=pool,
        max_retries=Retry(total=2, backoff_factor=0.2),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

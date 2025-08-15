class DummyResp:
    def __init__(self, data=None, status=200):
        self._data = data
        self.status = status

    async def json(self, content_type=None):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __await__(self):
        async def _self():
            return self
        return _self().__await__()


class DummySession:
    def __init__(self, data=None):
        self.data = data
        self.url = None
        self.params = None
        self.payload = None
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, params=None, timeout=10):
        self.url = url
        self.params = params
        return DummyResp(self.data)

    def post(self, url, json=None, timeout=10):
        self.url = url
        self.payload = json
        return DummyResp(self.data)

    async def close(self):
        self.closed = True

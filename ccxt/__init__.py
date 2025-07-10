class Exchange:
    pass

class RequestTimeout(Exception):
    pass

class NetworkError(Exception):
    pass

class ExchangeError(Exception):
    pass

def binance(params=None):
    return Exchange()

def coinbase(params=None):
    return Exchange()

def kraken(params=None):
    return Exchange()

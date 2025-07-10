class random:
    class RandomState:
        def __init__(self, seed=None):
            self.seed = seed

        def rand(self, *shape):
            if not shape:
                return 0.0
            if len(shape) == 1:
                return [0.0] * shape[0]
            return [[0.0] * shape[1] for _ in range(shape[0])]


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        return [0.0] * shape
    if isinstance(shape, tuple) and len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    return [0.0]

def isnan(x):
    return x != x

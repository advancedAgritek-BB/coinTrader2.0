class DataFrame(dict):
    def __init__(self, data=None):
        data = data or {}
        super().__init__(data)
        self.data = {k: list(v) for k, v in data.items()}
        self.columns = list(self.data.keys())
        self.__dict__.update(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(next(iter(self.data.values()), []))

    @property
    def empty(self):
        return len(self) == 0

    class _ILoc:
        def __init__(self, outer):
            self.outer = outer

        def __getitem__(self, idx):
            if isinstance(idx, int) and idx < 0:
                idx = len(self.outer) + idx
            return {k: v[idx] for k, v in self.outer.data.items()}

    @property
    def iloc(self):
        return self._ILoc(self)

    def pct_change(self):
        return self

    def to_numpy(self):
        return [list(v) for v in zip(*self.data.values())] if self.data else []

    def dropna(self):
        return self

    def std(self):
        return 0.0

    def mean(self):
        return 0.0

    def cumsum(self):
        return self


class Series(list):
    def tail(self, n):
        return Series(self[-n:])

    def to_numpy(self):
        return list(self)

    def std(self):
        return 0.0

    def mean(self):
        return 0.0

    def cummax(self):
        return self


def isnan(x):
    return x != x


def isna(x):
    return isnan(x)

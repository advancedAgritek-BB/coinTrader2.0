from collections import deque
from crypto_bot.utils.eval_queue import compute_batches, build_priority_queue


def test_compute_batches_basic():
    items = ['a', 'b', 'c', 'd', 'e']
    batches = compute_batches(items, 2)
    assert batches == [['a', 'b'], ['c', 'd'], ['e']]


def test_compute_batches_invalid_size():
    try:
        compute_batches(['a'], 0)
    except ValueError:
        pass
    else:
        assert False, 'expected ValueError'


def test_build_priority_queue_sorting():
    data = [('A', 1.0), ('B', 5.0), ('C', 3.0)]
    q = build_priority_queue(data)
    assert isinstance(q, deque)
    assert list(q) == ['B', 'C', 'A']


def test_priority_queue_rotation():
    symbols = ['A', 'B', 'C']
    scores = [(s, i) for i, s in enumerate(symbols)]
    queue = deque()
    batch_size = 2

    if len(queue) < batch_size:
        queue.extend(build_priority_queue(scores))
    batch1 = [queue.popleft() for _ in range(batch_size)]
    if len(queue) < batch_size:
        queue.extend(build_priority_queue(scores))
    batch2 = [queue.popleft() for _ in range(batch_size)]

    assert batch1 == ['C', 'B']
    assert batch2 == ['A', 'C']

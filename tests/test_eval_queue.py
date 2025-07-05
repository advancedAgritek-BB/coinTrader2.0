from crypto_bot.utils.eval_queue import compute_batches


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

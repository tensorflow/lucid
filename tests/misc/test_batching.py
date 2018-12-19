from lucid.misc.batching import batch, chunk


def test_batch():
    xs = range(8)  # note we're batching an iterator, not a list
    assert list(batch(xs, 1, as_list=True)) == [[0],[1],[2],[3],[4],[5],[6],[7]]
    assert list(batch(xs, 2, as_list=True)) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert list(batch(xs, 3, as_list=True)) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_chunk():
    assert list(chunk('', 1))== ['']
    assert list(chunk('ab', 2)) == ['a', 'b']
    assert list(chunk('abc', 2)) == ['ab', 'c']

    xs = list(range(8))  # note we're batching a list, not an iterator
    assert list(chunk(xs, 2)) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert list(chunk(xs, 3)) == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert list(chunk(xs, 5)) == [[0, 1], [2, 3], [4, 5], [6], [7]]

    rs = range(1000)
    assert list(chunk(rs, 2)) == [range(500), range(500, 1000)]

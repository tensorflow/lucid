# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for turning long lists into lists of smaller lists.

`batch` takes an iterator (of unknown length) and produces an iterator yielding batches
of `max_batch_size`.

`chunk` takes a sequence (of known length) and produces an iterator yielding exactly
`number_of_chunks` chunks splitting the original sequence as evenly as possible.

Both functions yield the smaller lists as generators, unless explicitly requested as
lists using the optional `as_list` parameter. If you instead need the entire result as
a list, please call `list()` on the resulting generator as usual.

If you are using numpy arrays, please use their optimized function `np.array_split()`.
"""

from itertools import accumulate, chain, islice, repeat, tee
from numpy import ndarray


def batch(iterable, max_batch_size, as_list=False):
    """Given an iterator (of unknown length), produces an iterator yielding batches
    of at most `max_batch_size`."""
    assert max_batch_size > 0
    assert not isinstance(iterable, ndarray)

    result = _batch(iterable, max_batch_size)
    if as_list:
        result = map(list, result)
    return result


def _batch(iterable, max_batch_size):
    source_iterator = iter(iterable)
    while True:
        batch_iterator = islice(source_iterator, max_batch_size)
        first_batch_item = next(batch_iterator)  # Exit on StopIteration
        yield chain((first_batch_item,), batch_iterator)


def chunk(sequence, number_of_chunks):
    """Given a fixed-length list, return exactly `number_of_chunks` sub-lists.
    Distributes elements in them as evenly as possible, i.e. th `number_of_chunks`
    sub-lists' lengths will differ by at most 1."""
    assert number_of_chunks > 0
    assert not isinstance(sequence, ndarray)

    return _chunk(sequence, number_of_chunks)


# inspired via http://wordaligned.org/articles/slicing-a-list-evenly-with-python
def _chunk(sequence, number_of_chunks):
    length = len(sequence)
    size, remainder = divmod(length, number_of_chunks)

    # we spread the `remainder` extra elements over the first `remainder` subsequences
    widths = chain(repeat(size+1, remainder), repeat(size, number_of_chunks-remainder))

    # accumulated widths make for slicing indexes. Tee "copies" the  iterator, advancing
    # it once (`next(end)`) to get a window of length 2 over offsets
    offsets = accumulate(chain((0,), widths))
    begin, end = tee(offsets)
    next(end)  # Exit on StopIteration
    for slicing in map(slice, begin, end):
        yield sequence[slicing]

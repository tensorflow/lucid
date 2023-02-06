from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.misc.channel_reducer import ChannelReducer


def test_channel_reducer_trivial():
    array = np.zeros((10, 10, 10))
    for d in range(array.shape[-1]):
        array[:, :, d] = np.eye(10, 10)

    channel_reducer = ChannelReducer(n_components=2)
    channel_reducer.fit(array)
    reduced = channel_reducer.transform(array)

    assert reduced.shape == (10, 10, 2)
    # the hope here is that this was reduced to use only the first channel
    # the reduced matrix should be all 0s, but in some python / tf configurations
    # the reduced array as some spurios small values
    assert np.sum(reduced[:, :, 1]) <= 1.e-12

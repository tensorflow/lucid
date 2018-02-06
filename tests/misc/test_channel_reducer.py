from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from lucid.misc.channel_reducer import ChannelReducer


def test_channel_reducer_trivial():
  array = np.zeros((10,10,10), dtype=np.float32)
  for d in range(array.shape[-1]):
    array[:,:,d] = np.eye(10,10)

  channel_reducer = ChannelReducer(n_features=2)
  channel_reducer.fit(array)
  reduced = channel_reducer.transform(array)

  assert reduced.shape == (10,10,2)
  assert (reduced[:,:,0] == array[:,:,0]).all()
  assert (reduced[:,:,1] == 0).all()

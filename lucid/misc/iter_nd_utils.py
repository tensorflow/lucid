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

"""Helpers for doing gnerator/iterable style workflows in n-dimensions."""

import itertools
import types
from collections.abc import Iterable
import numpy as np


def recursive_enumerate_nd(it, stop_iter=None, prefix=()):
  """Recursively enumerate nested iterables with tuples n-dimenional indices.

  Arguments:
    it: object to be enumerated
    stop_iter: User defined funciton which can conditionally block further
      iteration. Defaults to allowing iteration.
    prefix: index prefix (not intended for end users)

  Yields:
    (tuple representing n-dimensional index, original iterator value)

  Example use:
    it = ((x+y for y in range(10) )
               for x in range(10) )
    recursive_enumerate_nd(it) # yields things like ((9,9), 18)

  Example stop_iter:
    stop_iter = lambda x: isinstance(x, np.ndarray) and len(x.shape) <= 3
    # this prevents iteration into the last three levels (eg. x,y,channels) of
    # a numpy ndarray

  """
  if stop_iter is None:
    stop_iter = lambda x: False

  for n, x in enumerate(it):
    n_ = prefix + (n,)
    if isinstance(x, Iterable) and (not stop_iter(x)):
      yield from recursive_enumerate_nd(x, stop_iter=stop_iter, prefix=n_)
    else:
      yield (n_, x)


def dict_to_ndarray(d):
  """Convert a dictionary representation of an array (keys as indices) into a ndarray.

  Args:
    d: dict to be converted.

  Converts a dictionary representation of a sparse array into a ndarray. Array
  shape infered from maximum indices. Entries default to zero if unfilled.

  Example:
    >>> dict_to_ndarray({(0,0) : 3, (1,1) : 7})
    [[3, 0],
     [0, 7]]

  """
  assert len(d), "Dictionary passed to dict_to_ndarray() must not be empty."
  inds = list(d.keys())
  ind_dims = len(inds[0])
  assert all(len(ind) == ind_dims for ind in inds)
  ind_shape = [max(ind[i]+1 for ind in inds) for i in range(ind_dims)]

  val0  = d[inds[0]]
  if isinstance(val0, np.ndarray):
    arr = np.zeros(ind_shape + list(val0.shape), dtype=val0.dtype)
  else:
    arr = np.zeros(ind_shape, dtype="float32")

  for ind, val in d.items():
    arr[ind] = val
  return arr


def batch_iter(it, batch_size=64):
  """Iterate through an iterable object in batches."""
  while True:
    batch = list(itertools.islice(it, batch_size))
    if not batch: break
    yield batch

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

import numpy as np
import tensorflow as tf

from lucid.optvis.param import lowres_tensor


def multi_interpolation_basis(n_objectives=6, n_interp_steps=5, width=128,
                              channels=3):
  """A paramaterization for interpolating between each pair of N objectives.

  Sometimes you want to interpolate between optimizing a bunch of objectives,
  in a paramaterization that encourages images to align.

  Args:
    n_objectives: number of objectives you want interpolate between
    n_interp_steps: number of interpolation steps
    width: width of intepolated images
    channel

  Returns:
    A [n_objectives, n_objectives, n_interp_steps, width, width, channel]
    shaped tensor, t, where the final [width, width, channel] should be
    seen as images, such that the following properties hold:

     t[a, b]    = t[b, a, ::-1]
     t[a, i, 0] = t[a, j, 0] for all i, j
     t[a, a, i] = t[a, a, j] for all i, j
     t[a, b, i] = t[b, a, -i] for all i

  """
  N, M, W, Ch = n_objectives, n_interp_steps, width, channels

  const_term = sum([lowres_tensor([W, W, Ch], [W//k, W//k, Ch])
                    for k in [1, 2, 4, 8]])
  const_term = tf.reshape(const_term, [1, 1, 1, W, W, Ch])

  example_interps = [
      sum([lowres_tensor([M, W, W, Ch], [2, W//k, W//k, Ch])
           for k in [1, 2, 4, 8]])
      for _ in range(N)]

  example_basis = []
  for n in range(N):
    col = []
    for m in range(N):
      interp = example_interps[n] + example_interps[m][::-1]
      col.append(interp)
    example_basis.append(col)

  interp_basis = []
  for n in range(N):
    col = [interp_basis[m][N-n][::-1] for m in range(n)]
    col.append(tf.zeros([M, W, W, 3]))
    for m in range(n+1, N):
      interp = sum([lowres_tensor([M, W, W, Ch], [M, W//k, W//k, Ch])
                    for k in [1, 2]])
      col.append(interp)
    interp_basis.append(col)

  basis = []
  for n in range(N):
    col_ex = tf.stack(example_basis[n])
    col_in = tf.stack(interp_basis[n])
    basis.append(col_ex + col_in)
  basis = tf.stack(basis)

  return basis + const_term

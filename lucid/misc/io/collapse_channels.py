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

"""Convert an "image" wtih n channels into 3 RGB channels."""


import math
import numpy as np

def hue_to_rgb(ang, warp=True):
  """Produce an RGB unit vector corresponding to a hue of a given angle."""
  ang = ang - 360*(ang//360)
  colors = np.asarray([
      [1,0,0],
      [1,1,0],
      [0,1,0],
      [0,1,1],
      [0,0,1],
      [1,0,1],
  ])
  colors = colors / np.linalg.norm(colors, axis=1, keepdims=True)
  R = 360 / len(colors)
  n = math.floor(ang / R)
  D = (ang - n*R) / R

  if warp:
    # warping the angle away from the primary colors (RGB)
    # helps make equally-spaced angles more visually distinguishable
    adj = lambda x: math.sin(x * math.pi / 2)
    if n % 2 == 0:
      D = adj(D)
    else:
      D = 1 - adj(1 - D)

  v = (1-D) * colors[n] + D * colors[(n+1) % len(colors)]
  return v / np.linalg.norm(v)


def sparse_channels_to_rgb(X):
  assert (X >= 0).all()

  K = X.shape[-1]

  rgb = 0
  for i in range(K):
    ang = 360 * i / K
    color = hue_to_rgb(ang)
    color = color[tuple(None for _ in range(len(X.shape)-1))]
    rgb += X[..., i, None] * color

  rgb += np.ones(X.shape[:-1])[..., None] * (X.sum(-1) - X.max(-1))[..., None]
  rgb /= 1e-4 + np.linalg.norm(rgb, axis=-1, keepdims=True)
  rgb *= np.linalg.norm(X, axis=-1, keepdims=True)

  return rgb


def collapse_channels(X):

  if (X < 0).any():
    X = np.concatenate([np.maximum(0, X), np.maximum(0, -X)], axis=-1)
  return sparse_channels_to_rgb(X)

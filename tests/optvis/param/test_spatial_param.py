from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
import tensorflow as tf
from lucid.optvis.param import spatial

def test_sample_bilinear():
  h, w = 2, 3
  img = np.float32(np.arange(6).reshape(h, w, 1))
  img = img[::-1]  # flip y to match OpenGL
  tests = [[   0,    0,    0],
           [   1,    1,    4],
           [ 0.5,  0.5,    2],
           [ 0.5,  0.0,  0.5],
           [ 0.0,  0.5,  1.5],
           [-1.0, -1.0,  5.0],
           [   w,    1,  3.0],
           [     w-0.5,   h-0.5,  2.5],
           [   2*w-0.5, 2*h-0.5,  2.5]]
  tests = np.float32(tests)
  uv = np.float32((tests[:,:2] + 0.5) / [w, h]) # normalize UVs
  expected = tests[:,2:]

  with tf.Session() as sess:
    output = spatial.sample_bilinear(img, uv).eval()
    assert np.abs(output - expected).max() < 1e-8

